# ignore_header_test
# Copyright 2023 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from torch.cuda.amp import GradScaler
from generate_dataset import generate_normalized_graphs
from modulus.models.meshgraphnet import MeshGraphNet
from modulus.launch.logging import PythonLogger
from modulus.launch.utils import load_checkpoint
import hydra
from omegaconf import DictConfig
import json
import time
import vtk


def denormalize(tensor, mean, stdv):
    """Denormalize a tensor given a mean and a standard deviation.
       denormalized_tensor = (tensor * stdv) + mean

    Arguments:
        tensor: tensor to denormalize
        mean: mean used for normalization
        stdv: standard deviation used for normalization

    Returns:
        denormalized tensor
    """
    return tensor * stdv + mean


class MGNRollout:
    def __init__(self, logger, cfg):
        """Performs the rollout phase on the geometry specified in
        'config.yaml' (testing.graph) and computes the error"""

        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        logger.info(f"Using {self.device} device")

        params = json.load(open("checkpoints/parameters.json"))

        norm_type = {"features": "normal", "labels": "normal"}
        graphs, params = generate_normalized_graphs(
            "raw_dataset/graphs/",
            norm_type,
            cfg.training.geometries,
            params["statistics"],
        )
        graph = graphs[list(graphs)[0]]

        infeat_nodes = graph.ndata["nfeatures"].shape[1] + 1
        infeat_edges = graph.edata["efeatures"].shape[1]
        nout = 2
        nodes_features = [
            "area",
            "tangent",
            "type",
            "T",
            "dip",
            "sysp",
            "resistance1",
            "capacitance",
            "resistance2",
            "loading",
        ]

        edges_features = ["rel_position", "distance", "type"]

        params["infeat_nodes"] = infeat_nodes
        params["infeat_edges"] = infeat_edges
        params["out_size"] = nout
        params["node_features"] = nodes_features
        params["edges_features"] = edges_features
        params["rate_noise"] = 100
        params["rate_noise_features"] = 1e-5
        params["stride"] = 5

        self.graphs = graphs

        # instantiate the model
        self.model = MeshGraphNet(
            params["infeat_nodes"],
            params["infeat_edges"],
            2,
            processor_size=cfg.architecture.processor_size,
            hidden_dim_node_encoder=cfg.architecture.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=cfg.architecture.hidden_dim_edge_encoder,
            hidden_dim_processor=cfg.architecture.hidden_dim_processor,
            hidden_dim_node_decoder=cfg.architecture.hidden_dim_node_decoder,
        )

        if cfg.performance.jit:
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        self.scaler = GradScaler()
        # enable eval mode
        self.model.eval()

        # load checkpoint
        _ = load_checkpoint(
            os.path.join(cfg.checkpoints.ckpt_path, cfg.checkpoints.ckpt_name),
            models=self.model,
            device=self.device,
            scaler=self.scaler,
        )

        self.params = params
        self.var_identifier = {"p": 0, "q": 1}

    def compute_average_branches(self, graph, flowrate):
        """
        Average flowrate over branch nodes

        Arguments:
            graph: DGL graph
            flowrate: 1D tensor containing nodal flow rate values

        """
        branch_id = graph.ndata["branch_id"].cpu().detach().numpy()
        bmax = np.max(branch_id)
        for i in range(bmax + 1):
            idxs = np.where(branch_id == i)[0]
            rflowrate = torch.mean(flowrate[idxs])
            flowrate[idxs] = rflowrate

    def predict(self, graph_name):
        """
        Perform rollout phase for a single graph in the dataset

        Arguments:
            graph_name: the graph name.

        """
        graph = self.graphs[graph_name]
        graph = graph.to(self.device)
        self.graph = graph

        ntimes = graph.ndata["pressure"].shape[-1]
        nnodes = graph.ndata["pressure"].shape[0]

        self.pred = torch.zeros((nnodes, 2, ntimes), device=self.device)
        self.exact = graph.ndata["nfeatures"][:, 0:2, :]
        # copy initial condition
        self.pred[:, 0:2, 0] = graph.ndata["nfeatures"][:, 0:2, 0]

        inmask = graph.ndata["inlet_mask"].bool()
        invar = graph.ndata["nfeatures"][:, :, 0].clone().squeeze()
        efeatures = graph.edata["efeatures"].squeeze()
        nnodes = inmask.shape[0]
        nf = torch.zeros((nnodes, 1), device=self.device)
        start = time.time()
        for i in range(ntimes - 1):
            # set loading variable (check original paper for reference)
            invar[:, -1] = graph.ndata["nfeatures"][:, -1, i]
            # we set the next flow rate at the inlet (boundary condition)
            nf[inmask, 0] = graph.ndata["nfeatures"][inmask, 1, i + 1]
            nfeatures = torch.cat((invar, nf), 1)
            pred = self.model(nfeatures, efeatures, graph).detach()
            invar[:, 0:2] += pred
            # we set the next flow rate at the inlet since that is known
            invar[inmask, 1] = graph.ndata["nfeatures"][inmask, 1, i + 1]
            # flow rate must be constant in branches
            self.compute_average_branches(graph, invar[:, 1])
            invar[inmask, 1] = graph.ndata["nfeatures"][inmask, 1, i + 1]

            self.pred[:, :, i + 1] = invar[:, 0:2]

        end = time.time()
        self.logger.info(f"Rollout took {end - start} seconds!")

    def denormalize(self):
        """
        Denormalize predicted and exact pressure and flow rate values. This
        function must be called after 'predict'.

        Arguments:
            graph_name: the graph name.

        """
        self.pred[:, 0, :] = denormalize(
            self.pred[:, 0, :],
            self.params["statistics"]["pressure"]["mean"],
            self.params["statistics"]["pressure"]["stdv"],
        )
        self.pred[:, 1, :] = denormalize(
            self.pred[:, 1, :],
            self.params["statistics"]["flowrate"]["mean"],
            self.params["statistics"]["flowrate"]["stdv"],
        )
        self.exact[:, 0, :] = denormalize(
            self.exact[:, 0, :],
            self.params["statistics"]["pressure"]["mean"],
            self.params["statistics"]["pressure"]["stdv"],
        )
        self.exact[:, 1, :] = denormalize(
            self.exact[:, 1, :],
            self.params["statistics"]["flowrate"]["mean"],
            self.params["statistics"]["flowrate"]["stdv"],
        )

    def compute_errors(self):
        """
        Compute errors in pressure and flow rate. This function must be called
        after 'predict' and 'denormalize'. The errors are computed as l2 errors
        at the branch nodes for all timesteps.

        """
        bm = torch.reshape(self.graph.ndata["branch_mask"], (-1, 1, 1))
        bm = bm.repeat(1, 2, self.pred.shape[2])
        diff = (self.pred - self.exact) * bm
        errs = torch.sum(torch.sum(diff**2, axis=0), axis=1)
        norm = torch.sum(torch.sum((self.exact * bm) ** 2, axis=0), axis=1)
        errs = errs / norm
        errs = torch.sqrt(errs)

        self.logger.info(f"Relative error in pressure: {errs[0] * 100}%")
        self.logger.info(f"Relative error in flowrate: {errs[1] * 100}%")

    def write_vtk_file(self, graph_name, outfile, outdir=".", vtkcombo=False):
        """
        Write vtk files (one per timestep) given a graph and a solution.
        The file can be opened in Paraview.

        Arguments:
            graph: A DGL graph.
            solution: Tuple containing two n x m tensors, where n is the number of
                    nodes and m the number of timesteps. The first tensor contains
                    the pressure solution, the second contains
                    the flow rate solution
            outfile (string): name of output file. It can be take value "solution" for
                              the gnn approximation, or "reference" for the ground
                              truth.
            outdir (string): directory where results should be stored

        """

        def write(polydata, filename):
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(os.path.join(outdir, filename))
            writer.SetInputData(polydata)
            writer.Write()

        graph = self.graphs[graph_name]
        ntimesteps = self.pred.shape[2]

        if outdir != "." and not os.path.exists(outdir):
            os.makedirs(outdir)

        points = graph.ndata["x"].detach().numpy()
        edges0 = graph.edges()[0].detach().numpy()
        edges1 = graph.edges()[1].detach().numpy()

        if vtkcombo:
            polydata = vtk.vtkPolyData()
            dt = float(graph.ndata["dt"][0, 0])

        for t in range(ntimesteps):
            types = np.argmax(graph.edata["type"].detach().numpy(), axis=1)
            p_edges = np.where(types < 2)[0]

            if not vtkcombo:
                polydata = vtk.vtkPolyData()

            # Add points
            point_vtk = vtk.vtkPoints()
            for point in points:
                point_vtk.InsertNextPoint(point)
            polydata.SetPoints(point_vtk)

            # Prepare to add lines (cells)
            lines = vtk.vtkCellArray()
            for index in p_edges:
                pt1 = edges0[index]
                pt2 = edges1[index]
                lines.InsertNextCell(2)
                lines.InsertCellPoint(pt1)
                lines.InsertCellPoint(pt2)
            polydata.SetLines(lines)

            # Add Point Data
            pressure_array = vtk.vtkFloatArray()
            if vtkcombo:
                pressure_array.SetName(f"pressure__{t * dt:.3f}")
            else:
                pressure_array.SetName("pressure")
            pressure_array.SetNumberOfComponents(1)

            flowrate_array = vtk.vtkFloatArray()
            if vtkcombo:
                flowrate_array.SetName(f"flowrate_{t * dt:.3f}")
            else:
                flowrate_array.SetName("flowrate")
            flowrate_array.SetNumberOfComponents(1)

            if outfile == "solution":
                for i in range(len(points)):
                    pressure_array.InsertNextValue(self.pred[i, 0, t].item())
                    flowrate_array.InsertNextValue(self.pred[i, 1, t].item())
            elif outfile == "reference":
                for i in range(len(points)):
                    pressure_array.InsertNextValue(self.exact[i, 0, t].item())
                    flowrate_array.InsertNextValue(self.exact[i, 1, t].item())
            else:
                raise ValueError("Solution type " + outfile + " is unknown.")

            polydata.GetPointData().AddArray(pressure_array)
            polydata.GetPointData().AddArray(flowrate_array)

            if not vtkcombo:
                write(polydata, f"{outfile}_{t:04d}.vtp")

        if vtkcombo:
            write(polydata, f"{outfile}.vtp")


@hydra.main(version_base=None, config_path=".", config_name="config")
def do_rollout(cfg: DictConfig):
    """
    Perform rollout phase.

    Arguments:
        cfg: Dictionary containing problem parameters.

    """
    logger = PythonLogger("main")
    logger.file_logging()
    logger.info("Rollout started...")
    rollout = MGNRollout(logger, cfg)
    rollout.predict(cfg.testing.graph)
    rollout.denormalize()
    rollout.compute_errors()
    # change idx to plot pressure and flowrate at a different point
    rollout.write_vtk_file(
        cfg.testing.graph, "solution", "simulation_results/", vtkcombo=True
    )
    rollout.write_vtk_file(
        cfg.testing.graph, "reference", "simulation_results/", vtkcombo=True
    )


"""
The main function perform the rollout phase on the geometry specified in
'config.yaml' (testing.graph) and computes the error.
"""
if __name__ == "__main__":
    do_rollout()

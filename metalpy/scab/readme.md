SimPEG Commons and Boosters
===========================

**SimPEG Commons and Boosters** (**SCAB**), also known as **Super Cat's Armageddon with Bugs**, is a collection of common tools for SimPEG.
SCAB provides basic utilities and extensions for SimPEG, including: parallelization, demagnitization support and more. 

SCAB was previously named **SimPEG Common Auto Batching**, which is now a specific module for parallelization in SCAB.

Installation
------------
SCAB is now a submodule in metalpy, which can be installed using pip:

    pip install metalpy


Usage
-----
### Auto parallelization

Run your SimPEG simulations in parallel with only few changes.

The basic idea is to divide the receivers of simulation into batches, and each batch is run on a computing unit.
The batches are run in parallel, and the results will be merged back to be same as the serial version.

<table>
  <tr>
    <th>Before</th>
    <th>After</th>
  </tr>
  <tr>
    <td><pre lang="python">
import ...
import ...

receiver_list = Point(
    receiver_points, 
    components=components
)
receiver_list = [receiver_list]

source_field = sources.SourceField(
    receiver_list=receiver_list, parameters=H
)
survey = survey.Survey(source_field)

simulation = simulation.Simulation3DIntegral(
    survey=survey,
    mesh=mesh,
    model_type="scalar",
    chiMap=model_map,
    ind_active=active_ind,
    store_sensitivities="forward_only",
)
    </pre></td>
    <td><pre lang="python">
import ...
import ...
**_from metalpy.scab import parallelized_**
**_from metalpy.mepa import Executor_**

receiver_list = Point(
    receiver_points, 
    components=components
)
receiver_list = [receiver_list]

source_field = sources.SourceField.**_parallel_**(
    receiver_list=receiver_list, parameters=H
)
survey = survey.Survey.**_parallel_**(source_field)

**_executor = DaskExecutor('tcp://scheduler.addr:8786')_**

simulation = simulation.Simulation3DIntegral.**_parallel_**(
    survey=survey,
    mesh=mesh,
    model_type="scalar",
    chiMap=model_map,
    ind_active=active_ind,
    store_sensitivities="forward_only",
    **_executor=executor_**,
)
    </pre></td>
  </tr>
</table>

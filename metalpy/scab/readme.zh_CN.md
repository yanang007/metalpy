SimPEG Commons and Boosters
===========================

**SimPEG Commons and Boosters** (**SCAB**)，你也可以叫它**Super Cat's Armageddon with Bugs**（**超级猫猫与Bug的终焉之战**）。
SCAB是关于SimPEG的通用工具包，提供许多针对SimPEG的实用功能与扩展，包括: 自动并行化，退磁支持以及其他常用功能。 

SCAB原名**SimPEG Common Auto Batching**，其现在已经成为了SCAB的自动并行化模块。

安装
------------
SCAB目前是metalpy的一个子模块，你可以使用pip安装它：

    pip install metalpy

用法
-----
### 自动并行化

只需要修改几行代码，就可以让你的SimPEG仿真在并行模式下运行。

基本原理是将接收器分成多个批次，每个批次在一个计算单元上运行。
批次并行运行，最后将结果合并返回。

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
    actInd=active_ind,
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
    actInd=active_ind,
    store_sensitivities="forward_only",
    **_executor=executor_**,
)
    </pre></td>
  </tr>
</table>

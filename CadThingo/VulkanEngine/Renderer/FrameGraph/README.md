## Frame Graph Design and Usage Documentation
---
### Overview  

The **FrameGraph** is a compiled **D**irectional **A**cyclic **G**raph (**DAG**) designed to address the complex problem of sychronization with vulkan between
distinct GPU passes, the graph consists of **GraphPass**'s each which primarily declare a Setup method which defines what resources the pass requires, and an Execute
method which declares the code the pass will run during graph execution, this structure of pass usage and passes thus forms the graph structure, with usage 
representing edges and passes nodes.  
Key synchronisation concerns include:  

- Automatic derivation of Image and Buffer memory barriers based or resource usage
- And timeline semaphores to ensure safe cross queue(sync + async) compute work with resources  
  
Another responsibilty of the **FrameGraph** is to automate the creation of descriptors for graph resources, these exist at 2 distinct levels,
the graph shared set - bound by all passes within the graph(always at set index 2), and pass sets - containing the resources bound by a specific pass(always at set index 1).
The graph owns the responsibility for creating the descriptor pool and descriptor sets and then writing resources into descriptor sets, it is the
responsibility of the graph owner to specify and provide the descriptor set layout for both pass and graph shared sets.

--- 
### **GraphPass** Usage and Documentation

Passes contain a few key elements: 
- A PassType Where Pass type is either
  - A Graphics Pass
  - A Compute Pass
  - A RayTraced Pass
  - Or a Transfer Pass(Strictly Gpu memory managment)
- PassSetup where resource relationships are declared, Including pass set usage, resource usage read and/or writes, describing the action undertaken with the resource, this includes enforcing
- PassExecute declares the code that will be executed by the FrameGraph during execution for that pass.  
      
  An example block of code that declares a new pass could look like the following
  
    ```csharp
    scope.AddPass("examplePassName", PassType.Compute, QueueClass.Graphics,
            b =>
            {
                b.UsePassSet(_examplePass.PassSet);
                b.Read(readBuffer0, ResourceUsage.StorageReadCompute, "readBuffer0");
                graphBuffer0 = b.Write(writeBuffer0, ResourceUsage.StorageWriteCompute, "writeBuffer0");
                graphBuffer1 = b.Write(writeBuffer1, ResourceUsage.StorageWriteCompute, "writeBuffer1");
                graphBuffer2 = b.Write(writeBuffer2, ResourceUsage.StorageWriteCompute, "writeBuffer2");
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                _example.Record(cmd, f.FrameIndex, f.Camera, res.PassSet));
  ```



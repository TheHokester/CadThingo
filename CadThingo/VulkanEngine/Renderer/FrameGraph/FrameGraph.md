## Frame Graph Design and Usage Documentation
### Overview
---
The **FrameGraph** is a compiled **D**irectional **A**cyclic **G**raph (**DAG**) designed to address the complex problem of sychronization with vulkan between
distinct GPU passes, the graph consists of **GraphPass**'s each which primarily declare a Setup method which defines what resources the pass requires, and an Execute
method which declares the code the pass will run during graph execution, this structure of pass usage and passes thus forms the graph structure, with usage 
representing edges and passes nodes.  
Key synchronisation concerns include:  

- Automatic derivation of Image and Buffer memory barriers based or resource usage
- And timeline semaphores to ensure safe cross queue(sync + async) compute work with resources  
  
Another responsibilty of the **FrameGraph** is to automate the creation of descriptors for graph resources, these exist at 2 distinct levels,
the graph shared set - bound by all passes within the graph, and pass sets - containing the resources bound by a specific pass.
The graph owns the responsibility for creating the descriptor pool and descriptor sets and then writing resources into descriptor sets, it is the
responsibility of the graph owner to specify and provide the descriptor set layout for both pass and graph shared sets.

--- 
### **GraphPass** Usage




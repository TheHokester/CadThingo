namespace CadThingo.VulkanEngine;
//component that listens to events
//Handles physics related behavior and responds to events
// public sealed unsafe class PhysicsComponent : Component, IEventListener
// {
//     private readonly EventSystem _events;
//     
//     public PhysicsComponent(EventSystem events)
//     {
//         _events = events;
//     }
//
//     protected override void OnInitialize() => _events.AddListener(this);
//     protected override void OnDestroy() => _events.RemoveListener(this);
//
//     public void OnEvent(Event evt)
//     {
//         if (evt is CollisionEvent col)
//         {
//             HandleCollision(col);
//         }
//     }
//
//     private void HandleCollision(CollisionEvent col)
//     {
//         //TODO: do whatever 
//     }
// }
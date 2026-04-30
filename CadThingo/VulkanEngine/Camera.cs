using CadThingo.VulkanEngine.Renderer;
using Silk.NET.Input;

namespace CadThingo.VulkanEngine;
using Vec3 = System.Numerics.Vector3;
using Vec4 = System.Numerics.Vector4;
using Quat = System.Numerics.Quaternion;
using Mat4 = System.Numerics.Matrix4x4;


public class Camera : IEventListener
{
    //spacial postiioning and orientation vectors
    private protected Vec3 position;//camera location in world coordinates
    private protected Vec3 front;//camera forward direction where camera is looking  
    private protected Vec3 up;//cameras local up direction (for roll control)
    private protected Vec3 right; // camera local right direction (perpendicular to forward and up)
    private protected Vec3 worldUp = new Vec3(0.0f, 1.0f, 0.0f); // global up direction (y axis)
    
    
    //rotation represented as euler angles
    //provides intuitive control while managing gimbal lock and mathematical complexity internally
    private protected float yaw;//horizontal rotation around world y axis
    private protected float pitch;//vertical rotation around world x axis
    private float roll;//roll rotation around camera world z axis
    
    //User interaction and behaviour parameters
    private float movementSpeed = 3f; //units per second for translation movement
    private float mouseSensitivity = 0.15f;//degrees of rotation per pixel of raw mouse delta
    private float zoom = 45.0f;//field of view control for perspective projection

    // Held-state for movement keys, driven by KeyPressEvent/KeyReleaseEvent.
    // Tick() reads these per frame so holding a key keeps moving — pure event
    // dispatch only fires on the press edge, which alone wouldn't sustain motion.
    private bool moveForward, moveBack, moveLeft, moveRight, moveUp, moveDown;
    
    
    //Internal coordinate system maintenance
    //ensures mathematical consistency when orientation changes
    void UpdateCameraVectors()
    {
        Vec3 newFront;
        newFront.X = (float)(Math.Cos(ToRadians(yaw)) * Math.Cos(ToRadians(pitch)));
        newFront.Y = (float)Math.Sin(ToRadians(pitch));
        newFront.Z = (float)(Math.Sin(ToRadians(yaw)) * Math.Cos(ToRadians(pitch)));
        front = Vec3.Normalize(newFront);
        right = Vec3.Normalize(Vec3.Cross(front, worldUp));
        up = Vec3.Normalize(Vec3.Cross(right, front));
    }

    private protected float ToRadians(float angle)
    {
        return (float)(Math.PI/180) * angle;
    }

    public Camera()
    {
        position = new Vec3(0.0f, 0.0f, 3.0f);
        up = new Vec3(0.0f, 1.0f, 0.0f);
        yaw = -90f;
        pitch = 0.0f;
        roll = 0.0f;
        Engine.EventBus.AddListener(this, EventCategory.Input);
        UpdateCameraVectors();
    }
    /// <summary>
    /// Returns the view matrix for the camera.
    /// </summary>
    /// <returns></returns>
    public Mat4 GetViewMatrix()
    {
        return Mat4.CreateLookAt(position, position + front, up);
    }
    
    /// <summary>
    /// Returns the projection matrix for the camera.
    /// </summary>
    /// <param name="aspectRatio">Camera aspect ratio</param>
    /// <param name="nearPlane">near plane distance</param>
    /// <param name="farPlane">Far plane distance</param>
    /// <returns></returns>
    public Mat4 GetProjectionMatrix(float aspectRatio, float nearPlane, float farPlane)
    
    {
        return Mat4.CreatePerspectiveFieldOfView(ToRadians(zoom), aspectRatio, nearPlane, farPlane);
    }
    /// <summary>
    /// Processes keyboard input and adjusts camera position.<br/>
    /// W for forward, S for backward, A for left, D for right, SPACE for up, LShift for down.
    /// </summary>
    /// <param name="direction">movement direction</param>
    /// <param name="deltaTime">time elapsed</param>
    /// <exception cref="ArgumentOutOfRangeException">Non existent enum value passed</exception>
    public void ProcessKeyboard(CameraMovement direction, float deltaTime)
    {
        float velocity = movementSpeed * deltaTime;

        switch (direction)
        {
            case CameraMovement.FORWARD:
                position += front * velocity;
                break;
            case CameraMovement.BACKWARD:
                position -= front * velocity;
                break;
            case CameraMovement.LEFT:
                position -= right * velocity;
                break;
            case CameraMovement.RIGHT:
                position += right * velocity;
                break;
            case CameraMovement.UP:
                position += up * velocity;
                break;
            case CameraMovement.DOWN:
                position -= up * velocity;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(direction), direction, null);
        }
    }
    /// <summary>
    /// Processes raw mouse delta (pixels since last frame) and rotates the camera.
    /// yaw/pitch accumulate — each frame adds its delta to the running orientation.
    /// Y axis is inverted so moving the mouse up pitches up.
    /// </summary>
    public void ProcessMouseMovement(float xOffset, float yOffset, bool constrainPitch = true)
    {
        yaw   += xOffset * mouseSensitivity;
        pitch -= yOffset * mouseSensitivity;

        if (constrainPitch)
        {
            pitch = Math.Clamp(pitch, -89.0f, 89.0f);
        }

        UpdateCameraVectors();
    }

    /// <summary>
    /// Per-frame movement application. Reads held-key flags maintained by
    /// OnEvent and translates the camera. Must be called once per Update tick
    /// with the real frame delta so motion is framerate-independent.
    /// </summary>
    public void Tick(float deltaSeconds)
    {
        if (moveForward) ProcessKeyboard(CameraMovement.FORWARD,  deltaSeconds);
        if (moveBack)    ProcessKeyboard(CameraMovement.BACKWARD, deltaSeconds);
        if (moveLeft)    ProcessKeyboard(CameraMovement.LEFT,     deltaSeconds);
        if (moveRight)   ProcessKeyboard(CameraMovement.RIGHT,    deltaSeconds);
        if (moveUp)      ProcessKeyboard(CameraMovement.UP,       deltaSeconds);
        if (moveDown)    ProcessKeyboard(CameraMovement.DOWN,     deltaSeconds);
    }
    /// <summary>
    /// Processes mouse scroll input and adjusts camera zoom.
    /// </summary>
    /// <param name="yOffset"></param>
    public void ProcessMouseScroll(float yOffset)
    {
        
    }

    public Frustum GetFrustum() => default;
    
    //--------------------------------------------------
    //property access methods for external systems
    //Provide controlled access to internal state without exposing implementation details (seperation of concerns)
    public Vec3 GetPosition() => position;
    public Vec3 GetFront() => front;
    float GetZoom() => zoom;

    public void OnEvent(Event evt)
    {
        switch (evt)
        {
            case KeyPressEvent kp:
                SetMovementKey((Key)kp.GetKeyCode, true);
                break;
            case KeyReleaseEvent kr:
                SetMovementKey((Key)kr.GetKeyCode, false);
                break;
            case MouseMoveEvent mm:
                ProcessMouseMovement(mm.GetX(), mm.GetY(), true);
                break;
            case MouseScrollEvent:
                // TODO: zoom on scroll
                break;
        }
    }

    private void SetMovementKey(Key key, bool down)
    {
        switch (key)
        {
            case Key.W:         moveForward = down; break;
            case Key.S:         moveBack    = down; break;
            case Key.A:         moveLeft    = down; break;
            case Key.D:         moveRight   = down; break;
            case Key.Space:     moveUp      = down; break;
            case Key.ShiftLeft: moveDown    = down; break;
        }
    }
}


public enum CameraMovement
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

public class ThirdPersonCamera : Camera
{
    private Vec3 targetPosition;//current world position of camera target
    private Vec3 targetForward;//direction from camera to target
    
    //Camera behaviour config parameters
    private float followDistance; //desired distance between camera and target
    private float followHeight; //height offset above the target
    private float followSmoothness; //interpolation factor for smooth camera movement (0 instant, 1 never)
    
    //occlusion avoidance and collision management
    //these parameters control how the camera responds to environmental obstacles
    private float minDistance; // minimum allowed follow distance
    private float rayCastDistance; // maximum distance for occlusion detection
    
    
    //internal computational state for smooth motion
    //used to manage the mathematical aspects of camera movement
    private Vec3 desiredPosition; //target position the camera wants to reach
    private Vec3 smoothDampVelocity; //velocity state for smooth damping interpolation algos
    
    //constructor with reasonable defaults
    public ThirdPersonCamera()
    {
        followDistance = 5.0f;//medium distance between camera and target
        followHeight = 2f;//height offset above the target
        followSmoothness = 0.125f;//smoothness factor for camera movement
        minDistance = 1.0f;//minimum distance for camera movement
        
    }
    
    //Core functionality methods for camera behaviour
    public void UpdatePosition(Vec3 targetPos, Vec3 targetFwd, float deltaTime)
    {
        //update target properties
        targetPosition = targetPos;
        targetForward = Vec3.Normalize(targetFwd);
        
        //calculate the desired camera position
        //Position the amera behind and above the character
        var offset = -targetForward * followDistance;
        offset.Y = followHeight;
        
        desiredPosition = targetPosition + offset;
        
        //smooth camera movement using exponential smoothing
        var t = 1 - MathF.Pow(followSmoothness, deltaTime * 60f);
        position = Lerp(position, desiredPosition, t);
        
        static Vec3 Lerp(Vec3 a, Vec3 b, float t) => a + (b - a) * t;
        
        //update the camera to look at the target
        front = Vec3.Normalize(targetPosition - position);
        
        //recalculate right and up vectors
        right = Vec3.Normalize(Vec3.Cross(front, worldUp));
        up = Vec3.Normalize(Vec3.Cross(right, front));
    }
    /// <summary>
    /// Ensures the FOV is clear of obstructions and ensures the camera is not too close to the target.
    /// </summary>
    /// <param name="scene">Scene does not yet exist</param>
    public void HandleOcclusion(Scene scene)
    {
        Ray ray;
        ray.origin = targetPosition;
        ray.direction = Vec3.Normalize(desiredPosition - targetPosition);
        
        //check for intersections with scene objects
        RayCastHit hit = default;
        if (scene.RayCast(ray, ref hit, Vec3.Distance(targetPosition, desiredPosition)))
        {
            //if there is an intersection, move the camera to the hit point, minus a small offset
            float offset = 0.2f;
            position = hit.point - (ray.direction * offset);
            
            //Ensure we dont get too close to the target
            float currentDistance = Vec3.Distance(position, targetPosition);
            if (currentDistance < minDistance)
            {
                position = targetPosition + ray.direction * minDistance;
            }
            //Update the camera to look at the target
            front = Vec3.Normalize(targetPosition - position);
            right = Vec3.Normalize(Vec3.Cross(front, worldUp));
            up = Vec3.Normalize(Vec3.Cross(right, front));
        }
        
    }

    public void Orbit(float horizontalAngle, float verticalAngle)
    {
        yaw += horizontalAngle;
        pitch += verticalAngle;
        
        //constrain pitch to avoid gimbal lock
        pitch = Math.Clamp(pitch, -89.0f, 89.0f);
        
        //Calculate the new camera position based on spherical coordinates
        float radius = followDistance;
        float yawRad = ToRadians(yaw);
        float pitchRad = ToRadians(pitch);
        
        //convert to cartesian
        Vec3 offset = new Vec3();
        offset.X = radius * (float)Math.Cos(pitchRad) * (float)Math.Cos(yawRad);
        offset.Y = (float)(radius * Math.Sin(pitchRad));
        offset.Z = (float)(radius * Math.Sin(yawRad) * Math.Cos(pitchRad));
        
        //set pos
        desiredPosition = targetPosition + offset;
        
        //update the camera to look at the target
        front = Vec3.Normalize(targetPosition - position);
        right = Vec3.Normalize(Vec3.Cross(front, worldUp));
        up = Vec3.Normalize(Vec3.Cross(right, front));
    }
    
    
    //Runtime configuration methods for dynamic behaviour
    public void SetFollowDistance(float distance) => followDistance = distance;
    public void SetFollowHeight(float height) => followHeight = height;
    public void SetFollowSmoothness(float smoothness) => followSmoothness = smoothness;
    
    

}
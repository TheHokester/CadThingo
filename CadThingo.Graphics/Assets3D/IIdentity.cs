namespace CadThingo.Graphics.Assets3D;

public interface IIdentity
{
    Guid Id { get; }
    string? Name { get; }
    string? Description { get; }
}
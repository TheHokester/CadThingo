using System.Text;
using System.Windows;

using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Input.StylusPlugIns;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Cadthingo.Assets3D.Geometry;
using CadThingo.Rendering;

namespace CadThingo;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    private Renderer renderer;
    private FrameBuffer buffer;

    public MainWindow()
    {
        InitializeComponent();
        this.buffer = new FrameBuffer((int)Height,(int)Width);
        CompositionTarget.Rendering += OnRender;
    }

    private void OnRender(object? sender, EventArgs e)
    {
        this.renderer = new Renderer(this.buffer);
        renderer.Render(new List<RenderTriangle>());
    }
}
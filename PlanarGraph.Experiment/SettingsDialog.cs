using System.Windows.Forms;

namespace PlanarGraph.Experiment
{
    public partial class SettingsDialog : Form
    {
        public SettingsDialog()
        {
            InitializeComponent();
        }

        public bool EnableCudafy
        {
            get { return checkBoxEnableCudafy.Checked; }
            set { checkBoxEnableCudafy.Checked = value; }
        }
    }
}
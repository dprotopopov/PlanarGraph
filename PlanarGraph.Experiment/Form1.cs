using System;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using PlanarGraph.Algorithm;
using PlanarGraph.Data;

namespace PlanarGraph.Experiment
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            Workspace = new Workspace();
            Experiments = new BindingList<Experiment>();
            propertyGrid1.SelectedObject = Workspace;
            dataGridView1.RowTemplate = new DataGridViewRow();
            dataGridView1.DataSource = Experiments;
            GammaAlgorithm = new GammaAlgorithm
            {
                WorkerBegin = GammaAlgorithmStartTimer,
                WorkerComplite = GammaAlgorithmStopTimer
            };
            MacLaneAlgorithm = new MacLaneAlgorithm
            {
                WorkerBegin = MacLaneAlgorithmStartTimer,
                WorkerComplite = MacLaneAlgorithmStopTimer
            };
        }

        private GammaAlgorithm GammaAlgorithm { get; set; }
        private MacLaneAlgorithm MacLaneAlgorithm { get; set; }
        public Workspace Workspace { get; set; }
        public BindingList<Experiment> Experiments { get; set; }

        public TimeSpan GammaAlgorithmTotalExecutionTime { get; set; }
        public DateTime GammaAlgorithmStartTime { get; set; }

        public TimeSpan MacLaneAlgorithmTotalExecutionTime { get; set; }
        public DateTime MacLaneAlgorithmStartTime { get; set; }

        public void GammaAlgorithmStartTimer()
        {
            GammaAlgorithmStartTime = DateTime.Now;
        }

        public void GammaAlgorithmStopTimer(bool result)
        {
            GammaAlgorithmTotalExecutionTime += DateTime.Now.Subtract(GammaAlgorithmStartTime);
        }

        public void MacLaneAlgorithmStartTimer()
        {
            MacLaneAlgorithmStartTime = DateTime.Now;
        }

        public void MacLaneAlgorithmStopTimer(bool result)
        {
            MacLaneAlgorithmTotalExecutionTime += DateTime.Now.Subtract(MacLaneAlgorithmStartTime);
        }

        private void runToolStripMenuItem_Click(object sender, EventArgs e)
        {
            for (int n = Workspace.MinNumberOfVertixes;
                n <= Workspace.MaxNumberOfVertixes;
                n += Workspace.StepNumberOfVertixes)
                for (int m = Workspace.MinNumberOfSegments;
                    m <= Workspace.MaxNumberOfSegments;
                    m += Workspace.StepNumberOfSegments)
                {
                    var experiment = new Experiment();
                    experiment.StartDateTime = DateTime.Now;
                    experiment.NumberOfTest = Workspace.NumberOfTest;
                    experiment.NumberOfVertixes = n;
                    experiment.NumberOfSegments = m;
                    GammaAlgorithmTotalExecutionTime = new TimeSpan(0);
                    MacLaneAlgorithmTotalExecutionTime = new TimeSpan(0);
                    bool areEqual = true;
                    for (long i = 0; i < Workspace.NumberOfTest; i++)
                    {
                        Graph graph = Graph.Random(n, m);
                        bool result1 = GammaAlgorithm.IsPlanar(graph);
                        bool result2 = MacLaneAlgorithm.IsPlanar(graph);
                        areEqual = areEqual && (result1 == result2);
                    }
                    experiment.ResultsAreEqual = areEqual;
                    experiment.GammaAlgorithmTotalExecutionTime = GammaAlgorithmTotalExecutionTime;
                    experiment.MacLaneAlgorithmTotalExecutionTime = MacLaneAlgorithmTotalExecutionTime;
                    experiment.EndDateTime = DateTime.Now;
                    Experiments.Add(experiment);
                    dataGridView1.DataSource = null;
                    dataGridView1.DataSource = Experiments;
                    dataGridView1.Refresh();
                }
        }

        private void saveAsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (saveFileDialog1.ShowDialog() != DialogResult.OK) return;
            using (StreamWriter writer = File.CreateText(saveFileDialog1.FileName))
            {
                PropertyInfo[] properties = typeof (Experiment).GetProperties();
                writer.WriteLine(string.Join(";", properties.Select(property => property.Name)));
                foreach (Experiment experiment in Experiments)

                    writer.WriteLine(string.Join(";",
                        properties.Select(property => property.GetValue(experiment, null).ToString())));
                writer.Close();
            }
        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Close();
        }
    }
}
using System;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using PlanarGraph.Algorithm;
using PlanarGraph.Data;
using PlanarGraph.Worker;

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
                WorkerComplite = GammaAlgorithmStopTimer,
                WorkerLog = WorkerLog
            };
            MacLaneAlgorithm = new MacLaneAlgorithm
            {
                WorkerBegin = MacLaneAlgorithmStartTimer,
                WorkerComplite = MacLaneAlgorithmStopTimer,
                WorkerLog = WorkerLog
            };
        }

        void WorkerLog(string text)
        {
            if (textBox1.InvokeRequired)
            {
                var d = new WorkerLog(WorkerLog);
                object[] objects = { text };
                Invoke(d, objects);
            }
            else
            {
                textBox1.AppendText(text);
                textBox1.AppendText(Environment.NewLine);
            }
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
            backgroundWorker1.RunWorkerAsync(this);
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

        private void AddExperiment(Experiment experiment)
        {
            if (dataGridView1.InvokeRequired)
            {
                var d = new AddExperimentDeligate(AddExperiment);
                object[] objects = {experiment};
                Invoke(d, objects);
            }
            else
            {
                Experiments.Add(experiment);
                dataGridView1.DataSource = null;
                dataGridView1.DataSource = Experiments;
                dataGridView1.Refresh();
            }
        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void backgroundWorker1_DoWork(object sender, DoWorkEventArgs e)
        {
            var This = e.Argument as Form1;
            for (int n = This.Workspace.MinNumberOfVertixes;
                n <= This.Workspace.MaxNumberOfVertixes;
                n += This.Workspace.StepNumberOfVertixes)
                for (int m = This.Workspace.MinNumberOfSegments;
                    m <= This.Workspace.MaxNumberOfSegments;
                    m += This.Workspace.StepNumberOfSegments)
                {
                    GammaAlgorithmTotalExecutionTime = new TimeSpan(0);
                    MacLaneAlgorithmTotalExecutionTime = new TimeSpan(0);
                    var experiment = new Experiment();
                    var graph = Graph.Random(n, m);
                    experiment.Graph = graph.ToString();
                    experiment.NumberOfVertixes = graph.Vertices.Count;
                    experiment.NumberOfSegments = graph.Count;
                    experiment.StartDateTime = DateTime.Now;
                    experiment.IsPlanarByGammaAlgorithm = GammaAlgorithm.IsPlanar(graph);
                    experiment.IsPlanarByMacLaneAlgorithm = MacLaneAlgorithm.IsPlanar(graph);
                    experiment.EndDateTime = DateTime.Now;
                    experiment.ResultsAreEqual = (experiment.IsPlanarByGammaAlgorithm ==
                                                  experiment.IsPlanarByMacLaneAlgorithm);
                    experiment.GammaAlgorithmExecutionTime = GammaAlgorithmTotalExecutionTime;
                    experiment.MacLaneAlgorithmExecutionTime = MacLaneAlgorithmTotalExecutionTime;
                    This.AddExperiment(experiment);
                }
        }

        private delegate void AddExperimentDeligate(Experiment experiment);
    }
}
using System;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using MyCudafy;
using MyLibrary.Worker;
using PlanarGraph.Algorithm;
using PlanarGraph.Data;

namespace PlanarGraph.Experiment
{
    public partial class Form1 : Form
    {
        private static readonly SettingsDialog SettingsDialog = new SettingsDialog();

        private readonly MatrixIO _dataGridViewManual = new MatrixIO
        {
            ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize,
            Dock = DockStyle.Fill,
            Name = "dataGridViewManual",
            RowTemplate = {Height = 20},
            TabIndex = 0
        };

        public Form1()
        {
            InitializeComponent();
            tabPageManual.Controls.Add(_dataGridViewManual);
            RandomWorkspace = new RandomWorkspace();
            Experiments = new BindingList<Experiment>();
            Tasks = new BindingList<Task>();
            LastTaskIndex = 0;
            propertyGrid1.SelectedObject = RandomWorkspace;
            dataGridViewTask.RowTemplate = new DataGridViewRow();
            dataGridViewTask.DataSource = Tasks;
            dataGridViewExperiment.RowTemplate = new DataGridViewRow();
            dataGridViewExperiment.DataSource = Experiments;
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
            Settings.EnableCudafy = SettingsDialog.EnableCudafy;
        }

        private GammaAlgorithm GammaAlgorithm { get; set; }
        private MacLaneAlgorithm MacLaneAlgorithm { get; set; }
        public RandomWorkspace RandomWorkspace { get; set; }
        public BindingList<Experiment> Experiments { get; set; }
        public BindingList<Task> Tasks { get; set; }

        public TimeSpan GammaAlgorithmTotalExecutionTime { get; set; }
        public DateTime GammaAlgorithmStartTime { get; set; }

        public TimeSpan MacLaneAlgorithmTotalExecutionTime { get; set; }
        public DateTime MacLaneAlgorithmStartTime { get; set; }
        private int LastTaskIndex { get; set; }

        private void WorkerLog(string text)
        {
            if (textBox1.InvokeRequired)
            {
                var d = new WorkerLog(WorkerLog);
                object[] objects = {text};
                Invoke(d, objects);
            }
            else
            {
                textBox1.AppendText(DateTime.Now.ToString(CultureInfo.InvariantCulture));
                textBox1.AppendText("\t");
                textBox1.AppendText(text);
                textBox1.AppendText(Environment.NewLine);
            }
        }

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
            backgroundWorkerRun.RunWorkerAsync(this);
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
            if (dataGridViewExperiment.InvokeRequired)
            {
                var d = new AddExperimentDeligate(AddExperiment);
                object[] objects = {experiment};
                Invoke(d, objects);
            }
            else
            {
                Experiments.Add(experiment);
                dataGridViewExperiment.DataSource = null;
                dataGridViewExperiment.DataSource = Experiments;
                dataGridViewExperiment.Refresh();
            }
        }

        private void AddTask(Task task)
        {
            if (dataGridViewTask.InvokeRequired)
            {
                var d = new AddTaskDeligate(AddTask);
                object[] objects = {task};
                Invoke(d, objects);
            }
            else
            {
                Tasks.Add(task);
                dataGridViewTask.DataSource = null;
                dataGridViewTask.DataSource = Tasks;
                dataGridViewTask.Refresh();
            }
        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void backgroundWorkerRun_DoWork(object sender, DoWorkEventArgs e)
        {
            var This = e.Argument as Form1;
            while (This.LastTaskIndex < This.Tasks.Count)
            {
                Task task = This.Tasks[This.LastTaskIndex++];
                WorkerLog("Эксперимент №" + This.LastTaskIndex);
                GammaAlgorithmTotalExecutionTime = new TimeSpan(0);
                MacLaneAlgorithmTotalExecutionTime = new TimeSpan(0);
                var graph = new Graph(task.Graph);
                var experiment = new Experiment
                {
                    Graph = graph.ToString(),
                    NumberOfVertixes = graph.Vertices.Count,
                    NumberOfSegments = graph.Count
                };
                experiment.StartDateTime = DateTime.Now;
                experiment.IsPlanarByMacLaneAlgorithm = MacLaneAlgorithm.IsPlanar(graph);
                experiment.IsPlanarByGammaAlgorithm = GammaAlgorithm.IsPlanar(graph);
                experiment.EndDateTime = DateTime.Now;
                experiment.ResultsAreEqual = (experiment.IsPlanarByGammaAlgorithm ==
                                              experiment.IsPlanarByMacLaneAlgorithm);
                experiment.GammaAlgorithmExecutionTime = GammaAlgorithmTotalExecutionTime;
                experiment.MacLaneAlgorithmExecutionTime = MacLaneAlgorithmTotalExecutionTime;
                This.AddExperiment(experiment);
            }
        }

        private void addToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (tabControlEnter.SelectedTab == tabPageRandom)
                for (int n = RandomWorkspace.MinNumberOfVertixes;
                    n <= RandomWorkspace.MaxNumberOfVertixes;
                    n += RandomWorkspace.StepNumberOfVertixes)
                    for (int m = RandomWorkspace.MinNumberOfSegments;
                        m <= RandomWorkspace.MaxNumberOfSegments;
                        m += RandomWorkspace.StepNumberOfSegments)
                    {
                        Graph graph = Graph.Random(n, m);
                        var task = new Task
                        {
                            Graph = graph.ToString(),
                            NumberOfVertixes = graph.Vertices.Count,
                            NumberOfSegments = graph.Count
                        };
                        AddTask(task);
                    }
            else if (tabControlEnter.SelectedTab == tabPageManual)
            {
                bool[,] matrix = _dataGridViewManual.TheData;
                int rows = matrix.GetUpperBound(0) + 1;
                int cols = matrix.GetUpperBound(1) + 1;
                var graph = new Graph();
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        if (matrix[i, j])
                            graph.Add(new Vertex(i), new Vertex(j));
                var task = new Task
                {
                    Graph = graph.ToString(),
                    NumberOfVertixes = graph.Vertices.Count,
                    NumberOfSegments = graph.Count
                };
                AddTask(task);
            }
        }

        private void settingsToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SettingsDialog.ShowDialog();
            Settings.EnableCudafy = SettingsDialog.EnableCudafy;
        }

        private delegate void AddExperimentDeligate(Experiment experiment);

        private delegate void AddTaskDeligate(Task task);
    }
}
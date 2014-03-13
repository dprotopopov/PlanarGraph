using System;
using System.Windows.Forms;
using PlanarGraph.Data;

namespace PlanarGraph.Experiment
{
    public class Experiment
    {
        public DateTime StartDateTime { get; set; }
        public DateTime EndDateTime { get; set; }
        public int NumberOfVertixes { get; set; }
        public int NumberOfSegments { get; set; }
        public TimeSpan GammaAlgorithmExecutionTime { get; set; }
        public TimeSpan MacLaneAlgorithmExecutionTime { get; set; }
        public bool ResultsAreEqual { get; set; }
        public bool IsPlanarByGammaAlgorithm { get; set; }
        public bool IsPlanarByMacLaneAlgorithm { get; set; }
        public string Graph { get; set; }
    }
}
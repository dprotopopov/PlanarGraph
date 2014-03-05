using System;

namespace PlanarGraph.Experiment
{
    public class Experiment
    {
        public DateTime StartDateTime { get; set; }
        public DateTime EndDateTime { get; set; }
        public int NumberOfTest { get; set; }
        public int NumberOfVertixes { get; set; }
        public int NumberOfSegments { get; set; }
        public TimeSpan GammaAlgorithmTotalExecutionTime { get; set; }
        public TimeSpan HopcroftTarjanAlgorithmTotalExecutionTime { get; set; }
        public bool ResultsAreEqual { get; set; }
    }
}
namespace PlanarGraph.Experiment
{
    public class Workspace
    {
        public Workspace()
        {
            NumberOfTest = 5;
            MinNumberOfVertixes = 10;
            MinNumberOfSegments = 10;
            MaxNumberOfVertixes = 20;
            MaxNumberOfSegments = 20;
            StepNumberOfVertixes = 5;
            StepNumberOfSegments = 5;
        }

        public int NumberOfTest { get; set; }
        public int MinNumberOfVertixes { get; set; }
        public int MinNumberOfSegments { get; set; }
        public int MaxNumberOfVertixes { get; set; }
        public int MaxNumberOfSegments { get; set; }
        public int StepNumberOfVertixes { get; set; }
        public int StepNumberOfSegments { get; set; }
    }
}
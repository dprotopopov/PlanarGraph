namespace PlanarGraph.Experiment
{
    public class RandomWorkspace
    {
        public RandomWorkspace()
        {
            MinNumberOfVertixes = 10;
            MinNumberOfSegments = 10;
            MaxNumberOfVertixes = 20;
            MaxNumberOfSegments = 20;
            StepNumberOfVertixes = 2;
            StepNumberOfSegments = 2;
        }

        public int MinNumberOfVertixes { get; set; }
        public int MinNumberOfSegments { get; set; }
        public int MaxNumberOfVertixes { get; set; }
        public int MaxNumberOfSegments { get; set; }
        public int StepNumberOfVertixes { get; set; }
        public int StepNumberOfSegments { get; set; }
    }
}
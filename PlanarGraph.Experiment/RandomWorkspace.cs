namespace PlanarGraph.Experiment
{
    public class RandomWorkspace
    {
        public RandomWorkspace()
        {
            MinNumberOfVertixes = 8;
            MinNumberOfSegments = 8;
            MaxNumberOfVertixes = 16;
            MaxNumberOfSegments = 16;
            StepNumberOfVertixes = 4;
            StepNumberOfSegments = 4;
        }

        public int MinNumberOfVertixes { get; set; }
        public int MinNumberOfSegments { get; set; }
        public int MaxNumberOfVertixes { get; set; }
        public int MaxNumberOfSegments { get; set; }
        public int StepNumberOfVertixes { get; set; }
        public int StepNumberOfSegments { get; set; }
    }
}
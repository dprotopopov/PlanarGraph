namespace PlanarGraph.Worker
{
    public interface IWorker
    {
        WorkerBegin WorkerBegin { get; set; }
        WorkerComplite WorkerComplite { get; set; }
        WorkerLog WorkerLog { get; set; }
    }
}
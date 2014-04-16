using MyLibrary.Worker;
using PlanarGraph.Data;

namespace PlanarGraph.Algorithm
{
    public interface IPlanarAlgorithm : IWorker
    {
        bool IsPlanar(Graph graph);
    }
}
using PlanarGraph.Data;
using PlanarGraph.Worker;

namespace PlanarGraph.Algorithm
{
    public interface IPlanarAlgorithm: IWorker
    {
        bool IsPlanar(Graph graph);
    }
}
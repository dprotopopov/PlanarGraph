using System.Collections.Generic;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    public class VertexComparer : IComparer<Vertex>
    {
        public int Compare(Vertex x, Vertex y)
        {
            return x.Id - y.Id;
        }
    }
}
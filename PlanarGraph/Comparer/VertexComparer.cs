using System.Collections.Generic;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    public class VertexComparer : IComparer<Vertex>, IEqualityComparer<Vertex>
    {
        public int Compare(Vertex x, Vertex y)
        {
            return x.Id - y.Id;
        }

        public bool Equals(Vertex x, Vertex y)
        {
            return x.Id == y.Id;
        }

        public int GetHashCode(Vertex obj)
        {
            return (int) ((obj.Id ^0xAAAAAAAA) ^ (~obj.Id >> 1) ^ (obj.Id<<1));
        }
    }
}
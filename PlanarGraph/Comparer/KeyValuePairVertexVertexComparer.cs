using System.Collections.Generic;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    public class KeyValuePairVertexVertexComparer :
        IComparer<KeyValuePair<Vertex, Vertex>>, IEqualityComparer<KeyValuePair<Vertex, Vertex>>
    {
        private static readonly VertexComparer VertexComparer = new VertexComparer();

        public int Compare(KeyValuePair<Vertex, Vertex> x,
            KeyValuePair<Vertex, Vertex> y)
        {
            int value = VertexComparer.Compare(x.Key, y.Key);
            return value != 0 ? value : VertexComparer.Compare(x.Value, y.Value);
        }

        public bool Equals(KeyValuePair<Vertex, Vertex> x, KeyValuePair<Vertex, Vertex> y)
        {
            return VertexComparer.Equals(x.Key, y.Key) && VertexComparer.Equals(x.Value, y.Value);
        }

        public int GetHashCode(KeyValuePair<Vertex, Vertex> obj)
        {
            return VertexComparer.GetHashCode(obj.Key) ^ (~VertexComparer.GetHashCode(obj.Key) << 1) ^
                   (VertexComparer.GetHashCode(obj.Key) >> 1);
        }
    }
}
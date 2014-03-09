using System.Collections.Generic;
using PlanarGraph.Collections;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    public class KeyValuePairVertexVertexComparer :
        IComparer<KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection>>
    {
        private static readonly VertexComparer VertexComparer = new VertexComparer();

        public int Compare(KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection> x,
            KeyValuePair<KeyValuePair<Vertex, Vertex>, PathCollection> y)
        {
            int value = VertexComparer.Compare(x.Key.Key, y.Key.Key);
            return value != 0 ? value : VertexComparer.Compare(x.Key.Value, y.Key.Value);
        }
    }
}
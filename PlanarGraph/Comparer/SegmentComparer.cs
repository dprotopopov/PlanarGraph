using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    public class SegmentComparer : IComparer<Segment>
    {
        private static readonly VertexComparer VertexComparer = new VertexComparer();

        public int Compare(Segment x, Segment y)
        {
            int value = x.Count - y.Count;
            if (value != 0) return value;
            List<Vertex> list1 = x.ToList();
            List<Vertex> list2 = y.ToList();
            list1.Sort(VertexComparer);
            list2.Sort(VertexComparer);

            return list1.Select((t, i) => VertexComparer.Compare(t, list2[i])).FirstOrDefault(compare => compare != 0);
        }
    }
}
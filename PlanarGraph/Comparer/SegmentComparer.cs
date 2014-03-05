using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    public class SegmentComparer : IComparer<Segment>
    {
        public SegmentComparer()
        {
            VertexComparer = new VertexComparer();
        }

        private VertexComparer VertexComparer { get; set; }

        public int Compare(Segment x, Segment y)
        {
            List<Vertex> list1 = x.ToList();
            List<Vertex> list2 = y.ToList();
            int value = list1.Count - list2.Count;
            if (value != 0) return value;
            list1.Sort(VertexComparer);
            list2.Sort(VertexComparer);

            return list1.Select((t, i) => VertexComparer.Compare(t, list2[i])).FirstOrDefault(compare => compare != 0);
        }
    }
}
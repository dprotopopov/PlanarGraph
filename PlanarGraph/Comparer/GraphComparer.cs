using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Collections;
using PlanarGraph.Data;

namespace PlanarGraph.Comparer
{
    internal class GraphComparer : IComparer<Graph>
    {
        private static readonly VertexComparer VertexComparer = new VertexComparer();
        public int Compare(Graph x, Graph y)
        {
            var value = x.Count() - y.Count;
            if (value != 0) return value;
            var children1 = x.Children;
            var children2 = y.Children;
            value = children1.Count() - children2.Count;
            if (value != 0) return value;
            var list1 = new VertexSortedCollection(children1.Keys);
            var list2 = new VertexSortedCollection(children2.Keys);
            list1.Sort(VertexComparer);
            list2.Sort(VertexComparer);
            value = list1.Count - list2.Count;
            if (value != 0) return value;
            int count = list1.Count;
            if (count == 0) return 0;
            value = list1.Select((item, i) => VertexComparer.Compare(item, list2[i])).FirstOrDefault(v => v != 0);
            if (value != 0) return value;
            for (int k = 0; k < count; k++)
            {
                var list11 = new VertexSortedCollection(children1[list1[k]]);
                var list22 = new VertexSortedCollection(children2[list2[k]]);
                value = list11.Count() - list22.Count;
                if (value != 0) return value;
                list11.Sort(VertexComparer);
                list22.Sort(VertexComparer);
                value = list11.Select((item, i) => VertexComparer.Compare(item, list22[i])).FirstOrDefault(v => v != 0);
                if (value != 0) return value;
            }
            return 0;
        }
    }
}
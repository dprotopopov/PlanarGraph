using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Comparer;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    public class VertexCollection : StackListQueue<Vertex>
    {
        public VertexCollection(IEnumerable<int> list)
        {
            Comparer = new VertexComparer();
            AddRange(list.Select(id => new Vertex(id)));
        }

        public VertexCollection()
        {
            Comparer = new VertexComparer();
        }

        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public override string ToString()
        {
            return string.Join(",", this.Select(item => item.ToString()));
        }
    }
}
using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    public class VertexUnsortedCollection : StackListQueue<Vertex>
    {
        public VertexUnsortedCollection(IEnumerable<int> list)
        {
            AddRange(list.Select(id => new Vertex(id)));
        }

        public VertexUnsortedCollection()
        {
        }

        protected VertexUnsortedCollection(IEnumerable<Vertex> list)
        {
            AddRange(list);
        }

        public VertexUnsortedCollection(Vertex vertex)
            : base(vertex)
        {
        }

        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public override StackListQueue<int> GetInts(Vertex values)
        {
            return new StackListQueue<int> {values.Id};
        }

        public override string ToString()
        {
            return string.Join(",", this.Select(item => item.ToString()));
        }
    }
}
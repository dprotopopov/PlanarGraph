using System.Collections.Generic;
using System.Linq;
using MyCudafy.Collections;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    public class VertexUnsortedCollection : MyLibrary.Collections.StackListQueue<Vertex>
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

        public new int IndexOf(Vertex item)
        {
            return base.IndexOf(item);
        }

        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public override MyLibrary.Collections.StackListQueue<int> GetInts(Vertex values)
        {
            return new MyLibrary.Collections.StackListQueue<int> { values.Id };
        }

        public override string ToString()
        {
            return string.Join(",", this.Select(item => item.ToString()));
        }
    }
}
using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Collections;
using Enumerable = System.Linq.Enumerable;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс сегмента графа
    ///     Сегмент содержит две вершины
    /// </summary>
    public class Segment : VertexSortedCollection, IElement
    {
        public Segment(Vertex vertex1, Vertex vertex2) : base(vertex1)
        {
            Add(vertex2);
        }

        public Segment(IEnumerable<Vertex> list) : base(list)
        {
        }

        public override bool Equals(object obj)
        {
            var segment = obj as Segment;
            return segment != null && base.Equals(segment);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        /// <summary>
        ///     Если все контактные вершины сегмента S имеют номера вершин какой-то грани Γ,
        ///     то мы будем говорить, что грань Γ вмещает этот сегмент и обозначать S⊂Γ
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        public bool BelongsTo(Edge edge)
        {
            return edge.Contains(Enumerable.First(this)) &&
                   edge.Contains(Enumerable.Last(this));
        }

        public bool BelongsTo(Path path)
        {
            throw new System.NotImplementedException();
        }

        public bool BelongsTo(Segment segment)
        {
            throw new System.NotImplementedException();
        }

        public bool Contains(Graph graph)
        {
            throw new System.NotImplementedException();
        }

        public bool Contains(Circle circle)
        {
            throw new System.NotImplementedException();
        }

        public bool Contains(Edge edge)
        {
            throw new System.NotImplementedException();
        }

        public bool Contains(Path path)
        {
            throw new System.NotImplementedException();
        }

        public bool Contains(Segment segment)
        {
            throw new System.NotImplementedException();
        }

        public bool BelongsTo(Graph graph)
        {
            return graph.Contains(this);
        }

        public bool BelongsTo(Circle circle)
        {
            return circle.Contains(this);
        }

        public override string ToString()
        {
            return string.Format("({0})", base.ToString());
        }

        public bool ConnectedTo(Graph graph)
        {
            return graph.Vertices.Contains(this.First()) || graph.Vertices.Contains(this.Last());
        }
    }
}
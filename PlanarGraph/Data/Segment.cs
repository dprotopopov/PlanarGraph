using System.Collections.Generic;
using PlanarGraph.Collections;
using Enumerable = System.Linq.Enumerable;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс сегмента графа
    ///     Сегмент содержит две вершины
    /// </summary>
    public class Segment : VertexCollection
    {
        public Segment(Vertex vertex1, Vertex vertex2)
        {
            Add(vertex1);
            Add(vertex2);
        }

        public Segment(IEnumerable<Vertex> list)
        {
            AddRange(list);
        }

        public override bool Equals(object obj)
        {
            return base.Equals(obj);
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
        public bool Belongs(Edge edge)
        {
            return edge.Contains(Enumerable.First(this)) &&
                   edge.Contains(Enumerable.Last(this));
        }

        public override string ToString()
        {
            return string.Format("({0})", base.ToString());
        }
    }
}
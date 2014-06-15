using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MyCudafy.Collections;
using PlanarGraph.Collections;
using PlanarGraph.Comparer;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс грани графа
    ///     Грань описывается списком вершин, принадлежащим этой грани
    ///     Грань — это часть плоскости.
    /// </summary>
    public class Edge : VertexUnsortedCollection, IElement
    {
        private static readonly VertexComparer VertexComparer = new VertexComparer();

        public override MyLibrary.Collections.StackListQueue<int> GetInts(Vertex values)
        {
            return new MyLibrary.Collections.StackListQueue<int> { values.Id };
        }
        public Edge(IEnumerable<Vertex> list)
            : base(list)
        {
        }

        public Edge(Vertex vertex) : base(vertex)
        {
        }

        public bool BelongsTo(Graph graph)
        {
            throw new NotImplementedException();
        }

        public bool BelongsTo(Circle circle)
        {
            throw new NotImplementedException();
        }

        public bool BelongsTo(Edge edge)
        {
            throw new NotImplementedException();
        }

        public bool BelongsTo(Path path)
        {
            throw new NotImplementedException();
        }

        public bool BelongsTo(Segment segment)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Graph graph)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Circle circle)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Edge edge)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Path path)
        {
            throw new NotImplementedException();
        }

        public bool Contains(Segment segment)
        {
            throw new NotImplementedException();
        }

        public bool FromTo(IEnumerable<Vertex> collection)
        {
            throw new NotImplementedException();
        }

        public bool FromOrTo(IEnumerable<Vertex> collection)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///     Разбиение грани путём, контакные вершины которого принадлежат данной грани
        /// </summary>
        /// <returns></returns>
        public IEnumerable<Edge> Split(Path path)
        {
            Debug.Assert(path.FromTo(this));
            var list = new StackListQueue<Edge>();
            if (path.Count() < 2) return list;
            int index1 = IndexOf(path.First());
            int index2 = IndexOf(path.Last());
            List<Vertex> vertexs = path.ToList();
            if (index1 == index2)
            {
                // Вырожденый случай когда путь представляет собой цикл                
                // и пересечение с грань происходит только в одной точке                
                if (path.Count > 3) list.Add(new Edge(path.GetRange(0, path.Count - 1)));
                list.Add(this);
            }
            else
            {
                if (index1 > index2)
                {
                    int t = index1;
                    index1 = index2;
                    index2 = t;
                    vertexs.Reverse();
                }
                List<Vertex> list1 = GetRange(0, index1);
                list1.AddRange(vertexs.GetRange(0, vertexs.Count - 1));
                list1.AddRange(GetRange(index2, this.Count() - index2));
                list.Add(new Edge(list1));
                vertexs.Reverse();
                List<Vertex> list2 = GetRange(index1, index2 - index1);
                list2.AddRange(vertexs.GetRange(0, path.Count() - 1));
                list.Add(new Edge(list2));
            }
            Debug.WriteLineIf(list.Any(), this + " split by " + path + " is " +
                                          string.Join(",", list.Select(item => item.ToString())));
            return list;
        }

        public override string ToString()
        {
            return string.Format("[{0}]", base.ToString());
        }

        public override bool Equals(object obj)
        {
            var edge = obj as Edge;
            if (obj == null) return false;
            if (Count != edge.Count()) return false;
            if (Count == 0) return true;

            var list1 = new SortedStackListQueue<Vertex>(this) {Comparer = VertexComparer};
            var list2 = new SortedStackListQueue<Vertex>(edge) {Comparer = VertexComparer};
            if (!list1.Equals(list2)) return false;

            int index = edge.IndexOf(this[0]);
            var edge1 = new Edge(edge);
            edge1.Rotate(index);
            if (this.SequenceEqual(edge1)) return true;
            edge1.Rotate();
            edge1.Reverse();
            return this.SequenceEqual(edge1);
        }

        public override int GetHashCode()
        {
            return new Graph(this).GetHashCode();
        }
    }
}
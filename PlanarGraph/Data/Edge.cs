using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс грани графа
    ///     Грань описывается списком вершин, принадлежащим этой грани
    ///     Грань — это часть плоскости, окруженная простым циклом и не содержащая внутри себя других элементов графа.
    /// </summary>
    public class Edge : Circle
    {
        public Edge(IEnumerable<Vertex> list)
        {
            AddRange(list);
        }

        public Edge(Circle circle)
        {
            AddRange(circle);
        }

        /// <summary>
        ///     Разбиение грани путём, контакные вершины которого принадлежат данной грани
        /// </summary>
        /// <returns></returns>
        public IEnumerable<Edge> Split(Path path)
        {
            Debug.Assert(path.Belongs(this));
            var list = new List<Edge>();
            int index1 = IndexOf(path.First());
            int index2 = IndexOf(path.Last());
            List<Vertex> vertexs = path.ToList();
            if (index1 == index2)
            {
                // Вырожденый случай когда путь представляет собой цикл                
                // и пересечение с грань происходит только в одной точке                
                list.Add(new Edge(path.GetRange(0, path.Count - 1)));
                List<Vertex> range = GetRange(0, index1);
                range.AddRange(vertexs.GetRange(0, vertexs.Count - 1));
                range.AddRange(GetRange(index2, this.Count() - index2));
                list.Add(new Edge(range));
                return list;
            }
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
            return list;
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
            return string.Format("[{0}]", base.ToString());
        }
    }
}
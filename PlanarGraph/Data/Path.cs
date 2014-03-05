using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using PlanarGraph.Collections;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс пути в графе
    ///     Путь содержит неограниченное число вершин
    /// </summary>
    public class Path : VertexCollection
    {
        public Path(IEnumerable<Vertex> list)
        {
            AddRange(list);
        }

        public Path()
        {
        }

        public Path(Vertex vertex)
        {
            Add(vertex);
        }

        public Path(Vertex vertex1, Vertex vertex2)
        {
            Add(vertex1);
            Add(vertex2);
        }

        public override string ToString()
        {
            return string.Format("<{0}>", base.ToString());
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
        ///     Определение 3. Две простые цепи, соединяющие х1 и х2, называются непересекающимися (или
        ///     вершинно-непересекающимися), если у них нет общих вершин, отличных от х1 и х2 (и, следовательно, нет общих ребер).
        /// </summary>
        /// <param name="path1"></param>
        /// <param name="path2"></param>
        /// <returns></returns>
        public static bool IsNonIntersected(Path path1, Path path2)
        {
            Debug.Assert(new List<Vertex>
            {
                path1.First(),
                path1.Last(),
                path2.First(),
                path2.Last(),
            }.Distinct().Count() == 2);

            IEnumerable<Vertex> intersect =
                path1.GetRange(1, path1.Count - 2)
                    .Intersect(path2.GetRange(1, path2.Count - 2));
            return !intersect.Any();
        }

        public bool Belongs(Edge edge)
        {
            return edge.Contains(this.First()) &&
                   edge.Contains(this.Last());
        }

        public bool Belongs(Graph graph)
        {
            int count = this.Count(vertex => graph.Vertices.Contains(vertex));
            if (!graph.ChildrenOrParents.ContainsKey(this[0])) return false;
            for (int i = 0; i < count - 1; i++)
            {
                if (!graph.ChildrenOrParents[this[i]].Contains(this[i + 1]))
                    return false;
            }
            return true;
        }

        public IEnumerable<Path> Split(Graph graph)
        {
            var list = new List<Path>();
            int count1 = this.Count(vertex => graph.Vertices.Contains(vertex));
            if (count1 == 0) return list;
            int count = Count;
            Vertex first = this.First(vertex => graph.Vertices.Contains(vertex));
            int index = IndexOf(first);
            var indexes = new List<int> {index++};
            for (int i = 0; i < count - 1; i++, index++)
            {
                if (indexes.Count%2 == 1 &&
                    !graph.ChildrenOrParents[this[(index + count - 1)%count]].Contains(this[(index)%count]))
                    indexes.Add(index);

                if (indexes.Count%2 == 0 && graph.Vertices.Contains(this[index%count]))
                    indexes.Add(index);
            }
            if (indexes.Count <= 2) return list;
            for (int i = 1; 2*i < indexes.Count; i++)
            {
                index = indexes[2*i - 1] - 1;
                var path = new Path();
                while (index <= indexes[2*i]) path.Add(this[index++%count]);
                list.Add(path);
            }
            return list;
        }
    }
}
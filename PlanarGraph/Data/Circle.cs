using System;
using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Collections;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс цикла в графе
    ///     Цикл содержит неограниченное число вершин
    /// </summary>
    public class Circle : VertexCollection
    {
        public Circle()
        {
        }

        public Circle(IEnumerable<Vertex> list)
        {
            AddRange(list);
        }

        public override string ToString()
        {
            return string.Format("[{0}]", base.ToString());
        }

        /// <summary>
        ///     В задаче построения плоского графа особую роль играют простые циклы. Простые циклы - это квазициклы, у которых
        ///     локальная степень вершин равна двум. Особая роль простых циклов объясняется тем, что границей грани в плоском
        ///     графе, как правило, является простой цикл. Мощность подмножества простых циклов в графе меньше мощности множества
        ///     квазициклов.
        /// </summary>
        public bool IsSimpleCircle()
        {
            return this.Distinct().Count() == Count;
        }

        /// <summary>
        ///     Определение 1. τ-циклом графа называется простой цикл, между двумя любыми несмежными вершинами которого в
        ///     соответствующем графе не существует маршрутов меньшей длины, чем маршруты, принадлежащие данному циклу.
        /// </summary>
        /// <param name="graph"></param>
        /// <returns></returns>
        public bool IsTauCircle(Graph graph)
        {
            Dictionary<KeyValuePair<Vertex, Vertex>, List<Path>> dictionary1 = graph.GetMinPaths(this);
            Dictionary<KeyValuePair<Vertex, Vertex>, List<Path>> dictionary2 = new Graph(this).GetMinPaths(this);
            return !dictionary1.SelectMany(
                pair =>
                    dictionary2.ContainsKey(pair.Key)
                        ? pair.Value.Where(
                            value => value.Count < dictionary2[pair.Key].First().Count)
                        : pair.Value).Any();
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
            for (int i = 0; i < count; i++,index++)
            {
                if (indexes.Count%2 == 1 &&
                    !graph.ChildrenOrParents[this[(index + count - 1)%count]].Contains(this[(index)%count]))
                    indexes.Add(index);

                if (indexes.Count%2 == 0 && graph.Vertices.Contains(this[index%count]))
                    indexes.Add(index);
            }
            if (indexes.Count <= 2) return list;
            indexes.Add(count + indexes[1]);
            for (int i = 1; 2*i < indexes.Count; i++)
            {
                index = indexes[2*i - 1] - 1;
                var path = new Path();
                while (index <= indexes[2*i]) path.Add(this[index++%count]);
                list.Add(path);
            }
            return list;
        }

        public bool Belongs(Graph graph)
        {
            int count = this.Count(vertex => graph.Vertices.Contains(vertex));
            if (count != Count) return false;
            Vertex first = this.First(vertex => graph.Vertices.Contains(vertex));
            int index = IndexOf(first);
            for (int i = 0; i < count; i++, index++)
            {
                if (!graph.ChildrenOrParents[this[(index + count - 1)%count]].Contains(this[(index)%count]))
                    return false;
            }
            return true;
        }
    }
}
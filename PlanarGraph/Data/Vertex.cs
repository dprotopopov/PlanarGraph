using System.Globalization;

namespace PlanarGraph.Data
{
    /// <summary>
    ///     Класс вершины графа
    ///     В качестве идентификатора вершины используется число типа long
    /// </summary>
    public class Vertex
    {
        public Vertex(int id)
        {
            Id = id;
        }

        public Vertex(Vertex vertex)
        {
            Id = vertex.Id;
        }

        public int Id { get; private set; }

        public override bool Equals(object obj)
        {
            var vertex = obj as Vertex;
            return vertex != null && Id.Equals(vertex.Id);
        }

        public override int GetHashCode()
        {
            return Id.GetHashCode();
        }

        public override string ToString()
        {
            return Id.ToString(CultureInfo.InvariantCulture);
        }
    }
}
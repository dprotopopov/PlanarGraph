using System.Collections.Generic;

namespace PlanarGraph.Data
{
    public interface IElement
    {
        bool BelongsTo(Graph graph);
        bool BelongsTo(Circle circle);
        bool BelongsTo(Edge edge);
        bool BelongsTo(Path path);
        bool BelongsTo(Segment segment);

        bool Contains(Graph graph);
        bool Contains(Circle circle);
        bool Contains(Edge edge);
        bool Contains(Path path);
        bool Contains(Segment segment);
        bool FromTo(IEnumerable<Vertex> collection);
        bool FromOrTo(IEnumerable<Vertex> collection);
    }
}
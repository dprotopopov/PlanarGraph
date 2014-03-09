using System.Collections.Generic;
using System.Linq;
using PlanarGraph.Comparer;
using PlanarGraph.Data;

namespace PlanarGraph.Collections
{
    public class SegmentCollection : SortedStackListQueue<Segment>
    {
        private static readonly SegmentComparer SegmentComparer = new SegmentComparer();

        public SegmentCollection(IEnumerable<Segment> segments)
        {
            Comparer = SegmentComparer;
            AddRange(segments.ToList());
        }

        public SegmentCollection()
        {
            Comparer = SegmentComparer;
        }

        public SegmentCollection(Segment segment)
        {
            Comparer = SegmentComparer;
            Add(segment);
        }

        public new void Add(Segment segment)
        {
            if (!segment.First().Equals(segment.Last())) base.AddExcept(segment);
        }

        public new void AddRange(IEnumerable<Segment> segments)
        {
            base.AddRangeExcept(segments.Where(segment => !segment.First().Equals(segment.Last())));
        }

        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
}
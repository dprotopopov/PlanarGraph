using Cudafy;

namespace PlanarGraph.Parallel
{
    /// <summary>
    ///     Класс поиска всех пар пересекающихся последовательностей из двух множеств последовательностей
    /// </summary>
    public static class CudafyFindSequencesIntersections
    {
        public const int MAXBUFFERSIZE = 32*1024;
        public const int MAXINDEXCOUNT = 512;
        [Cudafy]
        private static int[] indexes1 = new int[MAXINDEXCOUNT];
        [Cudafy]
        private static int[] indexes2 = new int[MAXINDEXCOUNT];
        [Cudafy]
        private static int[] sequences1 = new int[MAXBUFFERSIZE];
        [Cudafy]
        private static int[] sequences2 = new int[MAXBUFFERSIZE];
    }
}
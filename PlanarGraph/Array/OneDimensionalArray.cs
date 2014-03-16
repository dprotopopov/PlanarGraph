using System;
using System.Runtime.InteropServices;

namespace PlanarGraph.Array
{
    public class OneDimensionalArray<T>
    {
        private readonly T[] _oneDimensional;

        public OneDimensionalArray(T[] oneDimensional)
        {
            _oneDimensional = oneDimensional;
        }

        public T[,] ToTwoDimensional(int rows)
        {
            int columns = _oneDimensional.Length/rows;
            var twoDimensional = new T[rows, columns];
            Buffer.BlockCopy(_oneDimensional, 0, twoDimensional, 0, Buffer.ByteLength(_oneDimensional));
            return twoDimensional;
        }
    }
}
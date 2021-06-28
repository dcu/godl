package godl

import (
	"log"
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestDataLoader(t *testing.T) {
	testCases := []struct {
		desc  string
		x, y  tensor.Tensor
		opts  DataLoaderOpts
		loops int
	}{
		{
			desc: "Example 1",
			x: tensor.New(
				tensor.WithShape(6),
				tensor.WithBacking(
					tensor.Range(tensor.Float32, 0, 6),
				),
			),
			y: tensor.New(
				tensor.WithShape(6),
				tensor.WithBacking(
					tensor.Range(tensor.Float32, 0, 6),
				),
			),
			opts: DataLoaderOpts{
				Shuffle:   true,
				BatchSize: 2,
				Drop:      false,
			},
			loops: 2,
		},
		{
			desc: "Example 2",
			x: tensor.New(
				tensor.WithShape(6),
				tensor.WithBacking(
					tensor.Range(tensor.Float32, 0, 6),
				),
			),
			y: tensor.New(
				tensor.WithShape(6),
				tensor.WithBacking(
					tensor.Range(tensor.Float32, 0, 6),
				),
			),
			opts: DataLoaderOpts{
				Shuffle:   false,
				BatchSize: 4,
				Drop:      false,
			},
			loops: 2,
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			log.Printf("%v %v", tC.x, tC.y)

			dl := NewDataLoader(tC.x, tC.y, tC.opts)
			count := 0

			for i := 0; i < tC.loops; i++ {
				log.Printf("Loop %d", i)

				for dl.HasNext() {
					xVal, yVal := dl.Next()

					log.Printf("%v %v", xVal, yVal)

					for b := 0; b < tC.opts.BatchSize; b++ {
						xx, err := xVal.At(b)
						c.NoError(err)
						yy, err := yVal.At(b)
						c.NoError(err)

						c.Equal(xx, yy)
					}

					count++
				}

				dl.Reset()
			}

			if !dl.opts.Drop && tC.x.Shape()[0]%dl.opts.BatchSize > 0 {
				n := tC.x.Shape()[0] % dl.opts.BatchSize
				c.Equal(count, tC.loops*(tC.x.Shape()[0]+n)/tC.opts.BatchSize)
			} else {
				c.Equal(count, tC.loops*tC.x.Shape()[0]/tC.opts.BatchSize)
			}

			log.Printf("count: %v", count)
		})
	}
}

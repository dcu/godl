package image

import (
	"testing"

	"github.com/fogleman/gg"
	"github.com/stretchr/testify/require"
)

func TestAugmentation(t *testing.T) {
	testCases := []struct {
		desc    string
		filters []Filter
	}{
		{
			desc: "Example 1",
			filters: []Filter{
				WithCrop(6, 2, 55, 100),
				Either(
					WithRandomRotation(0.8, -15, 15),
					WithRandomShear(0.8, -15, 15),
				),
				WithRandomGaussianBlur(0.5, 0.2, 1.0),
				WithRandomErosion(0.3, 0.5, 1.0),
			},
		},
	}

	img, err := gg.LoadImage("zebra.png")
	if err != nil {
		panic(err)
	}

	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			a := NewAugmenter(tC.filters...)
			result := a.ApplyN(img, 10)

			c.Len(result, 10)

			for i, r := range result {
				c.NotEqual(img, r)

				_ = i
				// 	_ = gg.SavePNG(fmt.Sprintf("%d.png", i), r)
			}
		})
	}
}

package imageutils

import (
	"fmt"
	"image"
	"io/fs"
	"log"
	"math/rand"
	"path/filepath"
	"time"

	"github.com/anthonynsimon/bild/adjust"
	"github.com/anthonynsimon/bild/blur"
	"github.com/anthonynsimon/bild/effect"
	"github.com/cheggaaa/pb/v3"
	"github.com/dcu/resize"
	"github.com/fogleman/gg"
	"github.com/oliamb/cutter"
)

var (
	randGen = rand.New(rand.NewSource(time.Now().UnixNano()))
)

// Augmenter is in charge of augmenting the given images
type Augmenter struct {
	filters []Filter
}

// Filter represents a filter applied to an image
type Filter func(img image.Image) (image.Image, error)

// WithCrop crops a region of the image, normally it should be applied as first filter
func WithCrop(x, y, width, height int) Filter {
	return func(img image.Image) (image.Image, error) {
		newImg, err := cutter.Crop(img, cutter.Config{
			Width:  width,
			Height: height,
			Anchor: image.Point{x, y},
		})
		if err != nil {
			log.Panicf("failed to crop image with (%d,%d) %dx%d: %v", x, y, width, height, err)
		}

		return newImg, nil
	}
}

// WithResize applies the resize operation to the image
func WithResize(width, height int) Filter {
	return func(img image.Image) (image.Image, error) {
		img = resize.Resize(uint(width), uint(height), img, resize.Lanczos3)

		return img, nil
	}
}

// WithRandomRotation configures the random rotation applied to an image
// probability is a number between 0 and 1 indicating the probability to apply this filter
func WithRandomRotation(probability, fromRadians, toRadians float64) Filter {
	return func(img image.Image) (image.Image, error) {
		if rand.Float64() > probability {
			return img, nil
		}

		rotation := randomFloat64(fromRadians, toRadians)

		dx := img.Bounds().Dx()
		dy := img.Bounds().Dy()
		minX := img.Bounds().Min.X
		minY := img.Bounds().Min.Y

		g := gg.NewContext(dx, dy)
		// g.DrawImage(img, -minX, -minY)
		g.RotateAbout(gg.Radians(rotation), float64(dx/2), float64(dy/2))
		g.DrawImage(img, -minX, -minY)

		return g.Image(), nil
	}
}

// WithRandomShear configures the random shear applied to an image
// probability is a number between 0 and 1 indicating the probability to apply this filter
func WithRandomShear(probability, fromRadians, toRadians float64) Filter {
	return func(img image.Image) (image.Image, error) {
		if rand.Float64() > probability {
			return img, nil
		}

		shear := randomFloat64(fromRadians, toRadians)

		dx := img.Bounds().Dx()
		dy := img.Bounds().Dy()
		minX := img.Bounds().Min.X
		minY := img.Bounds().Min.Y

		g := gg.NewContext(dx, dy)
		g.DrawImage(img, -minX, -minY)
		g.RotateAbout(gg.Radians(shear), float64(dx/2), float64(dy/2))
		g.DrawImage(img, -minX, -minY)

		return g.Image(), nil
	}
}

// WithRandomSaturation configures the random saturation applied to an image
func WithRandomSaturation(probability, minSaturation float64, maxSaturation float64) Filter {
	return func(img image.Image) (image.Image, error) {
		if rand.Float64() > probability {
			return img, nil
		}

		b := randomFloat64(minSaturation, maxSaturation)

		return adjust.Saturation(img, b), nil
	}
}

// WithRandomBrightness configures the random brightness applied to an image
// probability is a number between 0 and 1 indicating the probability to apply this filter
func WithRandomBrightness(probability, minBrightness float64, maxBrightness float64) Filter {
	return func(img image.Image) (image.Image, error) {
		if rand.Float64() > probability {
			return img, nil
		}

		b := randomFloat64(minBrightness, maxBrightness)

		return adjust.Brightness(img, b), nil
	}
}

// WithRandomGamma configures the random gamma applied to an image
// probability is a number between 0 and 1 indicating the probability to apply this filter
func WithRandomGamma(probability, minGamma float64, maxGamma float64) Filter {
	return func(img image.Image) (image.Image, error) {
		if rand.Float64() > probability {
			return img, nil
		}

		b := randomFloat64(minGamma, maxGamma)

		return adjust.Gamma(img, b), nil
	}
}

// WithRandomContrast configures the random contrast applied to an image
// probability is a number between 0 and 1 indicating the probability to apply this filter
func WithRandomContrast(probability, minContrast float64, maxContrast float64) Filter {
	return func(img image.Image) (image.Image, error) {
		if rand.Float64() > probability {
			return img, nil
		}

		b := randomFloat64(minContrast, maxContrast)

		return adjust.Contrast(img, b), nil
	}
}

// WithRandomHue configures the random hue applied to an image
// probability is a number between 0 and 1 indicating the probability to apply this filter
func WithRandomHue(probability float64, minHue int, maxHue int) Filter {
	return func(img image.Image) (image.Image, error) {
		b := randomInt(minHue, maxHue)

		return adjust.Hue(img, b), nil
	}
}

// WithRandomBoxBlur configures the random gaussian blur applied to an image
func WithRandomBoxBlur(probability, minBoxBlur float64, maxBoxBlur float64) Filter {
	return func(img image.Image) (image.Image, error) {
		if rand.Float64() > probability {
			return img, nil
		}

		b := randomFloat64(minBoxBlur, maxBoxBlur)

		return blur.Box(img, b), nil
	}
}

// WithRandomGaussianBlur configures the random gaussian blur applied to an image
// probability is a number between 0 and 1 indicating the probability to apply this filter
func WithRandomGaussianBlur(probability, minGaussianBlur float64, maxGaussianBlur float64) Filter {
	return func(img image.Image) (image.Image, error) {
		if rand.Float64() > probability {
			return img, nil
		}

		b := randomFloat64(minGaussianBlur, maxGaussianBlur)

		return blur.Gaussian(img, b), nil
	}
}

// WithRandomDilation configures the random dilation applied to an image
// probability is a number between 0 and 1 indicating the probability to apply this filter
func WithRandomDilation(probability, minDilation float64, maxDilation float64) Filter {
	return func(img image.Image) (image.Image, error) {
		if rand.Float64() > probability {
			return img, nil
		}

		b := randomFloat64(minDilation, maxDilation)

		return effect.Dilate(img, b), nil
	}
}

// WithRandomErosion configures the random erosion applied to an image
// probability is a number between 0 and 1 indicating the probability to apply this filter
func WithRandomErosion(probability, minErosion float64, maxErosion float64) Filter {
	return func(img image.Image) (image.Image, error) {
		if rand.Float64() > probability {
			return img, nil
		}

		b := randomFloat64(minErosion, maxErosion)

		return effect.Erode(img, b), nil
	}
}

// Either applies either of the given filters
func Either(filters ...Filter) Filter {
	return func(img image.Image) (image.Image, error) {
		filter := filters[randGen.Intn(len(filters))]

		return filter(img)
	}
}

// NewAugmenter builds a new augmenter
func NewAugmenter(filters ...Filter) *Augmenter {
	return &Augmenter{
		filters: filters,
	}
}

// Apply applies the filters to the given image
func (aug *Augmenter) Apply(img image.Image) (image.Image, error) {
	result, err := aug.ApplyN(img, 1)
	if err != nil {
		return nil, err
	}

	return result[0], nil
}

// ApplyN applies the filters to the given image N times
func (aug *Augmenter) ApplyN(img image.Image, n int) ([]image.Image, error) {
	var err error

	result := make([]image.Image, n)
	for i := 0; i < n; i++ {
		newImg := img
		for _, filter := range aug.filters {
			newImg, err = filter(newImg)
			if err != nil {
				return nil, err
			}
		}

		result[i] = newImg
	}

	return result, nil
}

// ApplyToDir applies all the augmentations to the images found in the given directory
func (aug *Augmenter) ApplyToDir(inputDir string, nAugmentations int, outputObserver func(n int, origPath string, img image.Image)) error {
	filesToProcess := []string{}

	err := filepath.WalkDir(inputDir, func(path string, d fs.DirEntry, err error) error {
		if d.IsDir() {
			return nil
		}

		filesToProcess = append(filesToProcess, path)

		return nil
	})
	if err != nil {
		return err
	}
	if len(filesToProcess) == 0 {
		return fmt.Errorf("no files found to process")
	}

	bar := pb.New(len(filesToProcess))
	bar.SetTemplate(pb.Full)
	bar.SetMaxWidth(80)
	bar.Start()

	defer bar.Finish()

	for _, path := range filesToProcess {
		bar.Increment()

		img, err := gg.LoadImage(path)
		if err != nil {
			// do not fail completely if the file was not an image
			continue
		}

		augmentations, err := aug.ApplyN(img, nAugmentations)
		if err != nil {
			log.Printf("failed to apply filters to %v: %v", path, err)

			continue
		}

		// TODO: this can be parallelized
		for i, aimg := range augmentations {
			outputObserver(i, path, aimg)
		}
	}

	return nil
}

func randomFloat64(min, max float64) float64 {
	r := min + randGen.Float64()*(max-min)

	return r
}

func randomInt(min, max int) int {
	r := randGen.Intn(max-min+1) + min

	return r
}

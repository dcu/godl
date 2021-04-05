package imageutils

import (
	"fmt"
	"image"
	"os"

	"github.com/dcu/resize"

	_ "image/jpeg" // import for side effects
	_ "image/png"  // import for side effects
)

// LoadOpts contains options to load an image
type LoadOpts struct {
	TargetSize []uint
}

// Load loads an image from the given path
func Load(filePath string, opts LoadOpts) (image.Image, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}

	defer func() { _ = file.Close() }()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("decoding image %v: %w", filePath, err)
	}

	if len(opts.TargetSize) == 2 {
		img = resize.Resize(opts.TargetSize[0], opts.TargetSize[1], img, resize.Lanczos3)
	}

	return img, nil
}

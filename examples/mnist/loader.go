package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"gorgonia.org/tensor"
)

type Mode string
type ContentType string

const (
	ModeTrain Mode = "train"
	ModeTest  Mode = "test"
)

const (
	ContentTypeData   ContentType = "data"
	ContentTypeLabels ContentType = "labels"
)

var (
	fileNames = map[Mode]map[ContentType]string{
		ModeTrain: {
			ContentTypeData:   "train-images-idx3-ubyte.gz",
			ContentTypeLabels: "train-labels-idx1-ubyte.gz",
		},
		ModeTest: {
			ContentTypeData:   "t10k-images-idx3-ubyte.gz",
			ContentTypeLabels: "t10k-labels-idx1-ubyte.gz",
		},
	}
)

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
)

func Load(mode Mode, baseDir string) (inputs, targets tensor.Tensor, err error) {
	labelFile := filepath.Join(baseDir, fileNames[mode][ContentTypeLabels])
	dataFile := filepath.Join(baseDir, fileNames[mode][ContentTypeData])

	labelData, err := loadLabelFile(labelFile)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot load label data: %w", err)
	}

	imageData, err := loadDataFile(dataFile)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot load image data: %w", err)
	}

	return imageDataToTensor(imageData), labelDataToTensor(labelData), nil
}

func pixelWeight(px byte) float32 {
	retVal := float32(px)/255*0.9 + 0.1
	if retVal == 1.0 {
		return 0.999
	}

	return retVal
}

func imageDataToTensor(imageData [][]byte) tensor.Tensor {
	rows := len(imageData)
	cols := len(imageData[0])

	backing := make([]float32, 0, rows*cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < len(imageData[i]); j++ {
			backing = append(backing, pixelWeight(imageData[i][j]))
		}
	}

	return tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(backing))
}

func labelDataToTensor(labelData []uint8) tensor.Tensor {
	rows := len(labelData)
	cols := 10

	backing := make([]float32, 0, rows*cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if j == int(labelData[i]) {
				backing = append(backing, 0.9)
			} else {
				backing = append(backing, 0.1)
			}
		}
	}

	return tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(backing))
}

func loadLabelFile(labelFile string) ([]uint8, error) {
	f, err := os.Open(labelFile)
	if err != nil {
		return nil, err
	}

	defer func() { _ = f.Close() }()

	r, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}

	var (
		magic int32
		n     int32
	)

	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}

	if magic != labelMagic {
		return nil, os.ErrInvalid
	}

	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}

	labels := make([]uint8, n)
	for i := 0; i < int(n); i++ {
		var l uint8
		if err := binary.Read(r, binary.BigEndian, &l); err != nil {
			return nil, err
		}

		labels[i] = l
	}

	return labels, nil
}

func loadDataFile(dataFile string) ([][]byte, error) {
	f, err := os.Open(dataFile)
	if err != nil {
		return nil, err
	}

	defer func() { _ = f.Close() }()

	r, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}

	var (
		magic int32
		n     int32
		nrow  int32
		ncol  int32
	)

	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}

	if magic != imageMagic {
		return nil, err
	}

	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}

	if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
		return nil, err
	}

	if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
		return nil, err
	}

	imgs := make([][]byte, n)
	m := int(nrow * ncol)
	for i := 0; i < int(n); i++ {
		imgs[i] = make([]byte, m)
		m_, err := io.ReadFull(r, imgs[i])
		if err != nil {
			return nil, err
		}
		if m_ != int(m) {
			return nil, os.ErrInvalid
		}
	}

	return imgs, nil
}

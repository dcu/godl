package storage

import (
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"os"
)

// LoadNN1 opens the given file path
func LoadNN1(loader *Storage, filePath string) error {
	f, err := os.Open(filePath)
	if err != nil {
		return err
	}

	defer func() {
		_ = f.Close()
	}()

	ungzipper, err := gzip.NewReader(f)
	if err != nil {
		return err
	}

	defer func() {
		_ = ungzipper.Close()
	}()

	dec := gob.NewDecoder(ungzipper)

	version := 0
	weightsCount := 0

	if err = dec.Decode(&version); err != nil {
		return err
	}

	if err = dec.Decode(&loader.Cost); err != nil {
		return err
	}

	if err = dec.Decode(&weightsCount); err != nil {
		return err
	}

	for i := 0; i < weightsCount; i++ {
		weight := Weight{}
		if err = dec.Decode(&weight); err != nil {
			return err
		}

		loader.AddWeights(weight)
	}

	return nil
}

// SaveNN1 saves the model in the given path
func SaveNN1(path string, item Item) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}

	defer func() {
		_ = f.Close()
	}()

	gzipper := gzip.NewWriter(f)
	defer func() {
		_ = gzipper.Close()
	}()

	enc := gob.NewEncoder(gzipper)
	version := 0

	if err = enc.Encode(version); err != nil {
		return fmt.Errorf("encoding version %d: %w", version, err)
	}

	if err = enc.Encode(item.Cost); err != nil {
		return fmt.Errorf("encoding cost %v: %w", item.Cost, err)
	}

	if err = enc.Encode(len(item.Weights)); err != nil {
		return fmt.Errorf("encoding weights count: %w", err)
	}

	for _, w := range item.Weights {
		if err = enc.Encode(&w); err != nil {
			return fmt.Errorf("encoding learnable %s: %w", w.Name, err)
		}
	}

	return nil
}

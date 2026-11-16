# Style presets and sample photos

PicturaAI ships 14 style presets and 3 sample photos. Every one of them is in the
public domain or released under CC0, and each file's source is recorded below. This
replaces the old set, which included works still under copyright and images with no
known source.

- Preset data: [`data/styles.json`](../../../data/styles.json) (contract: `BUILD-PLAN.md` §4)
- Full-size artwork (server engine): `images/style_image/<key>.jpg`, long side ≤ 1600 px,
  JPEG q85, metadata stripped
- Web sizes: `frontend/assets/styles/<key>-512.jpg | -480.webp | -160.webp`, built by
  `scripts/build_style_assets.py`
- Sample photos: `frontend/assets/samples/{dog,city,lake}.jpg` + `credits.json`
- Quality check: [`contact-sheet.jpg`](contact-sheet.jpg), built by
  [`make_contact_sheet.py`](make_contact_sheet.py)

## How "public domain" was decided

A work was accepted only if **both** of these hold:

1. **EU:** the artist died before 1956, so the life + 70 years term has ended in every EU
   country (the latest case here, Matisse, died in 1954: public domain since 1 January 2025).
2. **US:** the work was published or exhibited before 1931, so any US term has expired.
   This matters for artists who died after 1925. The 1994 URAA restored US copyright in
   foreign works that were still protected at home in 1996, so a painting by such an artist
   that was first published after 1930 can still be protected in the US. (Hilma af Klint
   is the classic trap: died 1944, first shown in 1986, US copyright until 2047.) For artists
   who died before 1926 every US term has run out, unless the work was first published
   between 1978 and 2002. That does not apply to any work here.

The scans and photos themselves add no new copyright. Faithful reproductions of
two-dimensional public-domain art are not protected in the US (*Bridgeman v. Corel*, 1999)
or in the EU (Article 14 of Directive (EU) 2019/790). All sources are Wikimedia Commons
files whose licence field (`LicenseShortName`) is "Public domain" or "CC0". The licence is
copied into each entry's `license` field.

## The 14 presets

| Key | Artwork | Artist (died) | US: why public domain | EU: why public domain | Source (Commons) | Licence |
|---|---|---|---|---|---|---|
| `starry_night` | *The Starry Night*, 1889. MoMA, New York | Vincent van Gogh (1890) | Artist died 1890; no US term can remain | Life + 70 ended 1960 | [File page](https://commons.wikimedia.org/wiki/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg) | Public domain |
| `the_scream` | *The Scream*, 1893. National Museum, Oslo | Edvard Munch (1944) | Exhibited in Berlin in 1893 | Life + 70 ended 2014 | [File page](https://commons.wikimedia.org/wiki/File:Edvard_Munch,_1893,_The_Scream,_oil,_tempera_and_pastel_on_cardboard,_91_x_73_cm,_National_Gallery_of_Norway.jpg) | Public domain |
| `great_wave` | *The Great Wave off Kanagawa*, c. 1831 (modern recut impression, c. 1930) | Katsushika Hokusai (1849) | Print published c. 1831. The c. 1930 recut is a faithful copy that adds no new authorship | Life + 70 ended 1919 | [File page](https://commons.wikimedia.org/wiki/File:The_Great_Wave_off_Kanagawa.jpg) | Public domain |
| `the_shipwreck` | *The Wreck of a Transport Ship* (also called *The Shipwreck of the Minotaur*), c. 1810. Calouste Gulbenkian Museum, Lisbon | J. M. W. Turner (1851) | Artist died 1851; no US term can remain | Life + 70 ended 1921 | [File page](https://commons.wikimedia.org/wiki/File:The_wreck_of_a_transport_ship_by_Turner_(1810).JPG) (museum photo by Benoît Prieur, who also released it as CC0) | Public domain |
| `udnie` | *Udnie (Young American Girl, The Dance)*, 1913. Centre Pompidou, Paris | Francis Picabia (1953) | Shown at the 1913 Salon d'Automne, Paris | Life + 70 ended 2023 | [File page](https://commons.wikimedia.org/wiki/File:Francis_Picabia,_1913,_Udnie_(Young_American_Girl,_The_Dance),_oil_on_canvas,_290_x_300_cm,_Mus%C3%A9e_National_d%E2%80%99Art_Moderne,_Centre_Georges_Pompidou,_Paris..jpg) | Public domain |
| `improvisation_27` | *Improvisation 27 (Garden of Love II)*, 1912. Metropolitan Museum of Art | Wassily Kandinsky (1944) | Shown at the 1913 Armory Show in New York, where Alfred Stieglitz bought it. The Met releases the image as CC0 Open Access | Life + 70 ended 2014 | [File page](https://commons.wikimedia.org/wiki/File:Improvisation_27_(Garden_of_Love_II)_MET_DP236124.jpg) | CC0 |
| `blue_horses` | *The Large Blue Horses*, 1911. Walker Art Center, Minneapolis | Franz Marc (1916) | Artist died 1916; no US term can remain | Life + 70 ended 1986 | [File page](https://commons.wikimedia.org/wiki/File:Franz_Marc_-_Die_gro%C3%9Fen_blauen_Pferde_1911.jpg) | Public domain |
| `the_kiss` | *The Kiss*, 1907–1908. Belvedere, Vienna | Gustav Klimt (1918) | Shown at the 1908 Kunstschau in Vienna, where the Austrian state bought it | Life + 70 ended 1988 | [File page](https://commons.wikimedia.org/wiki/File:The_Kiss_-_Gustav_Klimt_-_Google_Cultural_Institute.jpg) | Public domain |
| `grande_jatte` | *A Sunday on La Grande Jatte*, 1884–1886. Art Institute of Chicago | Georges Seurat (1891) | Shown at the eighth Impressionist exhibition, Paris, 1886 | Life + 70 ended 1961 | [File page](https://commons.wikimedia.org/wiki/File:Georges_Seurat_-_A_Sunday_on_La_Grande_Jatte_--_1884_-_Google_Art_Project.jpg) | Public domain |
| `woman_with_a_hat` | *Woman with a Hat*, 1905. SFMOMA, San Francisco | Henri Matisse (1954) | Shown at the 1905 Salon d'Automne and reproduced in *L'Illustration* on 4 November 1905. The Commons page carries `PD-US-expired` | Life + 70 ended 2024 | [File page](https://commons.wikimedia.org/wiki/File:Matisse-Woman-with-a-Hat.jpg) | Public domain |
| `cafe_terrace` | *Café Terrace at Night*, 1888. Kröller-Müller Museum, Otterlo | Vincent van Gogh (1890) | Artist died 1890; no US term can remain | Life + 70 ended 1960 | [File page](https://commons.wikimedia.org/wiki/File:Vincent_Willem_van_Gogh_-_Cafe_Terrace_at_Night_(Yorck).jpg) | Public domain |
| `zodiac` | *Zodiac*, 1896. Colour lithograph | Alphonse Mucha (1939) | Published as a printed lithograph in 1896 | Life + 70 ended 2009 | [File page](https://commons.wikimedia.org/wiki/File:Alphonse_Mucha_-_Zodiac,_1869.jpg) (the "1869" in the file name is a typo on Commons; the page dates it 1896) | Public domain |
| `sloop_nassau` | *Sloop, Nassau*, 1899. Watercolour. Metropolitan Museum of Art | Winslow Homer (1910) | Artist died 1910; no US term can remain (in the Met's collection since 1910) | Life + 70 ended 1980 | [File page](https://commons.wikimedia.org/wiki/File:Winslow_Homer_-_Sloop,_Nassau_(1899).jpg) | Public domain |
| `splashed_ink` | *Splashed-Ink Landscape* (*Haboku sansui*), 1495. Tokyo National Museum | Sesshū Tōyō (1506) | Over 500 years old | Over 500 years old | [File page](https://commons.wikimedia.org/wiki/File:Sesshu_-_Haboku-Sansui.jpg) | Public domain |

Files were downloaded as Commons thumbnails, checked with Pillow, and resized to a long
side of 1600 px or less. They were re-encoded as fresh JPEGs with no EXIF, XMP or ICC data.
Paper or mount margins were cropped from *Zodiac* and *Improvisation 27*.

**Why these 14.** Each candidate was used to stylise all three sample photos with the
real model (`run_nst`, `style_weight=1.0`, 512 px). Kept: works with a clear, different
look (palette, brushwork, era, culture). Dropped: works that came out muddy or close to
another preset. The final range runs from blue/yellow swirls, orange Expressionism and
blue-and-white ukiyo-e through amber Romantic storm, grey Cubist planes, soft abstract
colour, purple-red Expressionism, gold leaf, pointillist dots, Fauvist teal and violet,
warm night café, Art Nouveau terracotta and teal watercolour washes, to grey ink wash.

Candidates that were evaluated and **not** used:

| Candidate | Why not |
|---|---|
| Monet, *Impression, Sunrise* (1872) | Result muddy and desaturated, weak style |
| Cézanne, *Mont Sainte-Victoire and Château Noir* (1904–06) | Muted blue-green, close to *Sloop, Nassau*; pre-1931 exhibition not confirmed |
| Kandinsky, *Composition VII* (1913) | Strong result, but I could not confirm publication before 1931 (artist died 1944, so a later first publication would mean US copyright). Replaced by *Improvisation 27*, which has a clear 1913 Armory Show record |
| Franz Marc, *The Fate of the Animals* (1913) | Strong result, but a second Marc; *The Large Blue Horses* is more distinctive |
| Pissarro, *Boulevard Montmartre at Night* (1897) | Weaker version of the blue/yellow night look of the two Van Goghs |
| Rousseau, *Tiger in a Tropical Storm* (1891) | Flat olive-green, muddy |
| Guo Xi, *Early Spring* (1072); Ma Yuan, *Water* album (Song dynasty) | Results are plain sepia tints, close to *The Wreck of a Transport Ship*. Sesshū's ink wash gives the only grey monochrome result |
| Mucha, *Job* poster (1896) | A cigarette advertisement, and a weaker result than *Zodiac*. Mucha, *The Seasons*: weak and patchy |
| Homer, *The Turtle Pound* (1898) | Washed out; *Sloop, Nassau* is stronger |
| Hiroshige, *Sudden Shower over Shin-Ōhashi* (1857) | Weak, and a second Japanese print next to Hokusai |
| Signac, *The Port of Saint-Tropez* (1901–02) | Pale pastel, and a second pointillist next to Seurat |
| Derain, *Charing Cross Bridge* and *Houses of Parliament* (1906) | Pre-1931 publication not confirmed (artist died 1954); low-resolution source |
| Kirchner, *Berlin Street Scene* (1913) | Pre-1931 exhibition not confirmed (artist died 1938); source is a scan from a 2008 catalogue |

## Sample photos ("Try a sample")

| File | Photo | Author | Licence | Source |
|---|---|---|---|---|
| `dog.jpg` | Golden Retriever in a forest | Johan Spaedtke | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Golden_Retriever_adult.jpg) |
| `city.jpg` | Herzog-Friedrich-Strasse in the old town of Innsbruck, Austria | Wilfredor | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Herzog-Friedrich-Stra%C3%9Fe_in_Innsbruck%27s_Historic_Old_Town,_Austria.jpg) |
| `lake.jpg` | Clouds mirrored in Lake Bohinj, Slovenia | Ales Krivec | CC0 | [Commons](https://commons.wikimedia.org/wiki/File:Clouds_mirrored_in_a_mountain_lake_(Unsplash).jpg) |

All three are 1600 × ~1068 px, JPEG q85, with EXIF orientation applied and all metadata
stripped. No identifiable person is the subject. The city street has a few distant passers-by.
CC0 needs no attribution, but the credits are kept in `credits.json` for the credits page.

## Removed presets

| Key | What it was | Why removed |
|---|---|---|
| `la_muse` | Pablo Picasso, *La Muse* (1935) | **In copyright.** Picasso died in 1973, so it is protected in the EU until the end of 2043. As a 1935 work, it is also still protected in the US |
| `rain_princess` | Leonid Afremov, *Rain Princess* | **In copyright.** Afremov died in 2019, so it is protected until 2089 |
| `aquarelle` | Watercolour image, artist "Unknown" | Unknown source and licence. Replaced in role by Homer's *Sloop, Nassau* |
| `chinese_style` | Ink-style image, artist "Traditional" | Unknown source and licence. Replaced in role by Sesshū's *Splashed-Ink Landscape* |
| `space` | Nebula image, artist "Digital" | Unknown source and licence (looks like modern digital art) |
| `hampson` | Illustration credited to "Hampson" | Unknown source and licence (likely a living illustrator) |
| `mountain` | Mountain image, artist "Nature" | Unknown source and licence |
| `paris` | Painting of a Paris street, artist "Photography" | Unknown source and licence |

The kept artworks (*The Starry Night*, *The Scream*, *The Great Wave*, Turner's shipwreck
and *Udnie*) were also re-sourced from documented Commons files. Two files were renamed to
match their keys: `The_Great_Wave_off_Kanagawa.jpg` became `great_wave.jpg`, and
`the_shipwreck_of_the_minotaur.jpg` became `the_shipwreck.jpg`.

## Rebuilding

```bash
venv/Scripts/python.exe scripts/build_style_assets.py           # web sizes (skips up-to-date files)
venv/Scripts/python.exe scripts/build_style_assets.py --check   # validate styles.json + outputs
venv/Scripts/python.exe docs/revamp/presets/make_contact_sheet.py   # stylise samples, ~10 min on CPU
```

To add a preset, put a ≤ 1600 px JPEG in `images/style_image/`, add an entry to
`data/styles.json`, and add a row to the table above with its US and EU reasoning.
Then run the build script.

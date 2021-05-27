import os
from PIL import Image, ImageFile
import numpy as np
import random
import lmdb
import sys
import six

import torch
from torch.utils import data
from torch.utils.data import sampler
from torchvision import transforms

if __name__ == "__main__":
  sys.path.insert(0, '.')

from lib.utils.labelmaps import get_vocabulary, labels2strs
from lib.utils import to_numpy

ImageFile.LOAD_TRUNCATED_IMAGES = True


from config import get_args
global_args = get_args(sys.argv[1:])

if global_args.run_on_remote:
  import moxing as mox

CHAR_SUBSTITUTION_TABLE = {
  ' ': '',
  '\u00b4': "'",
  '\u00c9': 'E',
  '\u00e9': 'e'
}

class LmdbDataset(data.Dataset):
  def __init__(self, root, voc_type, max_len, num_samples, transform=None, with_name=False):
    super(LmdbDataset, self).__init__()

    if global_args.run_on_remote:
      dataset_name = os.path.basename(root)
      data_cache_url = "/cache/%s" % dataset_name
      if not os.path.exists(data_cache_url):
        os.makedirs(data_cache_url)
      if mox.file.exists(root):
        mox.file.copy_parallel(root, data_cache_url)
      else:
        raise ValueError("%s not exists!" % root)

      self.lmdb_dir = data_cache_url
    else:
      self.lmdb_dir = root

    txn = self.__open_lmdb()
    self.voc_type = voc_type
    self.transform = transform
    self.max_len = max_len
    self.nSamples = int(txn.get(b"num-samples"))
    self.nSamples = min(self.nSamples, num_samples)

    assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    self.EOS = 'EOS'
    self.PADDING = 'PADDING'
    self.UNKNOWN = 'UNKNOWN'
    self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
    self.char2id = dict(zip(self.voc, range(len(self.voc))))
    self.id2char = dict(zip(range(len(self.voc)), self.voc))

    self.rec_num_classes = len(self.voc)
    self.lowercase = (voc_type == 'LOWERCASE')

    self.with_name = with_name
    self.some_labels_truncated = False

  def __open_lmdb(self):
    env = lmdb.open(self.lmdb_dir, max_readers=32, readonly=True, create=False)
    assert env is not None, "cannot create lmdb from %s" % self.lmdb_dir
    return env.begin()

  def __len__(self):
    return self.nSamples

  def __getitem__(self, index):
    if not hasattr(self, 'txn'):
      self.txn = self.__open_lmdb()

    assert index <= len(self), 'index range error'
    index += 1
    img_key = b'image-%09d' % index
    imgbuf = self.txn.get(img_key)

    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    try:
      img = Image.open(buf).convert('RGB')
    except IOError:
      print('Corrupted image for %d' % index)
      return self[index + 1]

    # reconition labels
    label_key = b'label-%09d' % index
    word = self.txn.get(label_key).decode()
    if self.lowercase:
      word = word.lower()
    ## fill with the padding token
    label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)
    label_list = []
    for char in word:
      if len(label_list) + 1 == self.max_len:
        if not self.some_labels_truncated:
          self.some_labels_truncated = True # Print it only once
          print(f'Some labels were truncated to {self.max_len} including the stop symbol.')
        break
      if char in self.char2id:
        label_list.append(self.char2id[char])
      else:
        sub_char = CHAR_SUBSTITUTION_TABLE.get(char)
        if sub_char is not None:
          if self.with_name:
            print(f'Substituting "\\u{ord(char):04x}" with "{sub_char}".')
        else:
          ## add the unknown token
          print(f'Character "\\u{ord(char):04x}" is out of vocabulary for {index}.')
          label_list.append(self.char2id[self.UNKNOWN])
    ## add a stop token
    label_list = label_list + [self.char2id[self.EOS]]
    assert len(label_list) <= self.max_len
    label[:len(label_list)] = np.array(label_list)

    if len(label) <= 0:
      return self[index + 1]

    # label length
    label_len = len(label_list)

    if self.transform is not None:
      img = self.transform(img)

    if self.with_name:
      return img, label, label_len, self.txn.get(b'name-%09d' % index)
    else:
      return img, label, label_len


class ResizeNormalize(object):
  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation
    self.toTensor = transforms.ToTensor()

  def __call__(self, img):
    img = img.resize(self.size, self.interpolation)
    img = self.toTensor(img)
    img.sub_(0.5).div_(0.5)
    return img


class RandomSequentialSampler(sampler.Sampler):

  def __init__(self, data_source, batch_size):
    self.num_samples = len(data_source)
    self.batch_size = batch_size

  def __len__(self):
    return self.num_samples

  def __iter__(self):
    n_batch = len(self) // self.batch_size
    tail = len(self) % self.batch_size
    index = torch.LongTensor(len(self)).fill_(0)
    for i in range(n_batch):
      random_start = random.randint(0, len(self) - self.batch_size)
      batch_index = random_start + torch.arange(0, self.batch_size)
      index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
    # deal with tail
    if tail:
      random_start = random.randint(0, len(self) - self.batch_size)
      tail_index = random_start + torch.arange(0, tail)
      index[(i + 1) * self.batch_size:] = tail_index

    return iter(index.tolist())


class AlignCollate(object):

  def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    self.imgH = imgH
    self.imgW = imgW
    self.keep_ratio = keep_ratio
    self.min_ratio = min_ratio

  def __call__(self, batch):
    images, labels, lengths = zip(*batch)
    b_lengths = torch.IntTensor(lengths)
    b_labels = torch.IntTensor(labels)

    imgH = self.imgH
    imgW = self.imgW
    if self.keep_ratio:
      ratios = []
      for image in images:
        w, h = image.size
        ratios.append(w / float(h))
      ratios.sort()
      max_ratio = ratios[-1]
      imgW = int(np.floor(max_ratio * imgH))
      imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
      imgW = min(imgW, 400)

    transform = ResizeNormalize((imgW, imgH))
    images = [transform(image) for image in images]
    b_images = torch.stack(images)

    return b_images, b_labels, b_lengths

class AlignCollateWithNames(AlignCollate):
  def __call__(self, batch):
    return AlignCollate.__call__(self, [item[:3] for item in batch]) + \
      ([item[3] for item in batch],)

def test():
  train_dataset = LmdbDataset(
    root=global_args.test_data_dir,
    voc_type=global_args.voc_type,
    max_len=global_args.max_len,
    num_samples=global_args.num_test,
    with_name=True)
  print(train_dataset.nSamples, 'samples')
  train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=global_args.workers,
        collate_fn=AlignCollateWithNames(imgH=64, imgW=256, keep_ratio=False))

  if global_args.image_path:
    out_html = open(os.path.join(global_args.image_path, 'index.html'), 'w')
    out_html.write('''<html>
<body>
<table>
<tr><th>No</th><th>Image</th><th>Labels</th><th>Length</th><th>Name</th></tr>
''')
  else:
    out_html = None

  i = 1
  max_len = 0
  for images, labels, label_lens, image_names in train_dataloader:
    # visualization of input image
    # toPILImage = transforms.ToPILImage()
    images = images.permute(0,2,3,1)
    images = to_numpy(images)
    images = images * 0.5 + 0.5
    images = images * 255
    for image, label, label_len, image_name in zip(images, labels, label_lens, image_names):
      image = Image.fromarray(np.uint8(image))
      label_str = labels2strs(label, train_dataset.id2char, train_dataset.char2id)
      if image_name is not None:
        image_name = image_name.decode('utf-8')
      else:
        image_name = ''
      l_len = label_len.item()
      if max_len < l_len:
        max_len = l_len

      if global_args.image_path:
        image_filename = f'image-{i:09d}.jpg'
        image.save(os.path.join(global_args.image_path, image_filename))
        out_html.write(
          f'<tr><td>{i}</td>'
          f'<td><img src="{image_filename}" width="{image.width}" height="{image.height}" /></td>'
          f'<td>{label_str}</td><td>{l_len}</td><td>{image_name}</td></tr>\n')
      else:
        image.show()
        print(image.size)
        print(label_str, l_len)
        if image_name:
          print(image_name)
        input()
      i += 1

  if out_html:
    out_html.write(f'</table>\n<p>The maximal label length is {max_len}.</p>\n</body>\n</html>\n')
    out_html.close()

if __name__ == "__main__":
  test()

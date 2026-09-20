"""BPE 分词器快速测试"""
import sys
sys.path.insert(0, '..')
from tokenizer import BPETokenizer, CharTokenizer

# 测试文本（重复多次以产生足够统计）
text = ("The quick brown fox jumps over the lazy dog. "
        "To be or not to be, that is the question. "
        "All that glitters is not gold. "
        "The only thing we have to fear is fear itself. "
        "Ask not what your country can do for you. "
        "I have a dream that one day this nation will rise up. ") * 20

print('训练 BPE 分词器...')
tok = BPETokenizer.train(text, vocab_size=300)
print(f'BPE: {tok}')

# 编码/解码
test_str = 'The only thing'
ids = tok.encode(test_str)
decoded = tok.decode(ids)
print(f'  编码: "{test_str}" -> {ids}')
print(f'  解码: {ids} -> "{decoded}"')
assert decoded == test_str, f'编解码不一致: "{decoded}" != "{test_str}"'
print('  ✓ 编解码往返')

# 对比字符级
tok_char = CharTokenizer.train(text)
ids_char = tok_char.encode(test_str)
print(f'  字符级: {len(ids_char)} tokens, BPE: {len(ids)} tokens')
print(f'  压缩比: {len(ids) / len(ids_char):.2f}x')

# 保存/加载
import os
os.makedirs('../test_tmp', exist_ok=True)
tok.save('../test_tmp/tokenizer.json')
tok2 = BPETokenizer.load('../test_tmp/tokenizer.json')
ids2 = tok2.encode(test_str)
assert ids2 == ids, '保存/加载后编码不一致'
print('  ✓ 保存/加载往返')

# 清理
import shutil
shutil.rmtree('../test_tmp', ignore_errors=True)

print()
print('✓ BPE TOKENIZER TEST: PASSED')

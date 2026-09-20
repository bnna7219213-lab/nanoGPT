"""BPE 调试脚本 — 测试重写后的分词器"""
import sys
sys.path.insert(0, '..')
from tokenizer import BPETokenizer, CharTokenizer

text = "The cat sat on the mat. The dog ran in the park." * 5

tok = BPETokenizer.train(text, vocab_size=300)
print(f"BPE: {tok}")
print(f"合并数: {len(tok.merges)}")

# 打印前 5 个 merge 的规则
print("\n前 5 个 merges (按 id 排序):")
sorted_merges = sorted(tok.merges.items(), key=lambda x: x[1])
for (id1, id2), new_id in sorted_merges[:5]:
    print(f"  ({id1}, {id2}) -> {new_id}")

# 编码/解码测试
for test in ["The", "The cat sat", "Hello", "xyz123"]:
    ids = tok.encode(test)
    decoded = tok.decode(ids)
    ok = "✓" if decoded == test else "✗"
    print(f"  {ok} '{test}' -> {ids} -> '{decoded}'")

# 保存/加载测试
import os
os.makedirs('test_tmp', exist_ok=True)
tok.save('test_tmp/tok.json')
tok2 = BPETokenizer.load('test_tmp/tok.json')
assert tok2.encode("The cat") == tok.encode("The cat")
print("  ✓ 保存/加载往返")

import shutil
shutil.rmtree('test_tmp', ignore_errors=True)

# 对比
tok_c = CharTokenizer.train(text)
print(f"\n对比: BPE vocab={tok.vocab_size} vs Char vocab={tok_c.vocab_size}")
print(f"  BPE 编码 'The cat': {tok.encode('The cat')} ({len(tok.encode('The cat'))} tokens)")
print(f"  Char 编码 'The cat': {tok_c.encode('The cat')} ({len(tok_c.encode('The cat'))} tokens)")

print("\n✓ BPE TEST PASSED")

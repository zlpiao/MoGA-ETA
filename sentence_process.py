import re
import random

def split_and_mask(text):
    # 分割主句和原因部分
    main_clause, reasons_part = re.split(r'\bbecause\b', text, maxsplit=1, flags=re.IGNORECASE)
    main_clause = main_clause.strip() + " because"
    reasons_part = reasons_part.strip()

    # 精准分句逻辑：匹配句号、逗号+and、或独立逗号（排除分词短语）
    clauses = re.split(r'(?<=\.[^\w])|(?<=\.[^\w])|,\s+and\s+|\s*,\s+', reasons_part)
    clauses = [c.strip() for c in clauses if c.strip()]

    # 过滤非独立子句（如分词短语）
    independent_clauses = []
    for clause in clauses:
        if re.match(r'^(the|there|they|this)', clause, re.IGNORECASE) and clause.count(' ') > 4:
            independent_clauses.append(clause)
    if not independent_clauses:
        independent_clauses = clauses

    # 随机保留至少一个子句
    num_to_keep = random.randint(1, min(3,len(independent_clauses)))
    kept_clauses = random.sample(independent_clauses, num_to_keep)

    # 修复标点和语法
    combined = ""
    for i, clause in enumerate(kept_clauses):
        # 首字母大写处理
        clause = clause[0].lower() + clause[1:]
        # 连接符处理
        if i == 0:
            combined += clause.rstrip('.')
        else:
            if kept_clauses[i-1].endswith('.'):
                combined += " " + clause.rstrip('.')
            else:
                combined += "; " + clause.rstrip('.')
    # 确保以句号结尾
    if not combined.endswith('.'):
        combined += '.'

    # 合并主句
    final_sentence = f"{main_clause} {combined}"
    # 清理多余空格
    final_sentence = re.sub(r'\s+([.,])', r'\1', final_sentence)
    return final_sentence
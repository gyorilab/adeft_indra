from lxml import etree

def get_abbreviations(xml):
    try:
        tree = etree.fromstring(xml)
    except etree.XMLSyntaxError:
        return {}
    items = tree.xpath('.//glossary/def-list/def-item')
    result = {}
    for item in items:
        term_elements = item.xpath('./term')
        if not term_elements:
            continue
        term = ''.join(term_elements[0].itertext()).strip()
        def_elements = item.xpath('./def')
        if not def_elements:
            continue
        def_ = ''.join(def_elements[0].itertext()).strip()
        result[term] = def_
    return result
